from genericpath import isdir
from services.arguments.arguments_service_base import ArgumentsServiceBase
from entities.cache.cache_options import CacheOptions
import os
from services.log_service import LogService
import urllib.request
import random
from shutil import copyfile
from multiprocessing import Pool, TimeoutError
import functools
import sys
import pickle

from typing import Callable, List

from enums.language import Language

from services.data_service import DataService
from services.string_process_service import StringProcessService
from services.cache_service import CacheService


class OCRDownloadService:
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            data_service: DataService,
            string_process_service: StringProcessService,
            cache_service: CacheService,
            log_service: LogService):
        self._data_service = data_service
        self._string_process_service = string_process_service
        self._cache_service = cache_service
        self._log_service = log_service
        self._arguments_service = arguments_service

        self._languages_2017 = [
            Language.English,
            Language.French
        ]

        self._datasets = arguments_service.datasets

    def get_downloaded_file_paths(self, language: Language) -> List[str]:
        folder_paths = []
        prefixes = self._get_folder_language_prefixes(language)

        if language in self._languages_2017:
            folder_paths_2017 = [os.path.join('data', 'newseye', '2017', 'full', prefix) for prefix in prefixes]
            folder_paths.extend([x for x in folder_paths_2017 if os.path.exists(x)])

        folder_paths_2019 = [os.path.join('data', 'newseye', '2019', 'full', prefix) for prefix in prefixes]
        folder_paths.extend([x for x in folder_paths_2019 if os.path.exists(x)])

        result = []
        for folder_path in folder_paths:
            if not os.path.isdir(folder_path):
                result.append(folder_path)
                continue

            inner_level_names = os.listdir(folder_path)
            folder_paths.extend([os.path.join(folder_path, x) for x in inner_level_names])

        return result

    def download_data(self, language: Language, max_string_length: int = None):
        key_length_suffix = ''
        if max_string_length is not None:
            key_length_suffix = f'-{max_string_length}'

        if language in self._languages_2017:
            self._download_dataset(
                'icdar-2017',
                f'icdar-2017{key_length_suffix}',
                extraction_function=lambda: self.process_newseye_files(
                    language,
                    os.path.join('data', 'newseye', '2017'),
                    max_string_length=max_string_length))

        self._download_dataset(
            'icdar-2019',
            f'icdar-2019{key_length_suffix}',
            extraction_function=lambda: self.process_newseye_files(
                language,
                os.path.join('data', 'newseye', '2019'),
                max_string_length=max_string_length))

        if language == Language.English:
            self._download_dataset(
                'trove',
                'trove',
                extraction_function=self._process_trove_data)

    def get_downloaded_dataset(
        self,
        dataset:str,
        max_string_length: int = None):
        cache_key = dataset
        if dataset != 'trove' and max_string_length is not None:
            cache_key = f'{dataset}-{max_string_length}'

        return self._cache_service.get_item_from_cache(
            CacheOptions(
                cache_key,
                configuration_specific=False))

    def _download_dataset(
            self,
            dataset: str,
            dataset_cache_key: str,
            extraction_function: Callable):
        # we skip this dataset if it is not one of the used ones for the current run
        if dataset not in self._datasets:
            return

        cache_options = CacheOptions(
            dataset_cache_key,
            configuration_specific=False)

        # if the dataset was already processed before we do not need to do anything else
        if self._cache_service.item_exists(cache_options):
            return

        self._log_service.log_debug(f'Processing "{dataset}" dataset...')

        # get the data from the extraction callable function and cache it
        dataset_data = extraction_function()
        self._cache_service.cache_item(dataset_data, cache_options)

    def _cut_string(
            self,
            text: str,
            chunk_length: int):
        invalid_characters = ['#', '@']

        if chunk_length is None:
            return self._string_process_service.remove_string_characters(text, characters=invalid_characters)

        string_chunks = [
            self._string_process_service.convert_string_unicode_symbols(
                self._string_process_service.remove_string_characters(
                    text=text[i:i+chunk_length],
                    characters=invalid_characters))
            for i in range(0, len(text), chunk_length)]

        return string_chunks

    def process_newseye_files(
            self,
            language: Language,
            data_path: str,
            start_position: int = 14,
            max_string_length: int = 500,
            subfolder_to_use: str = 'full'):
        ocr_sequences = []
        gs_sequences = []

        language_prefixes = self._get_folder_language_prefixes(language)

        for subdir_name in os.listdir(data_path):
            if subdir_name != subfolder_to_use:
                continue

            subdir_path = os.path.join(data_path, subdir_name)
            if not os.path.isdir(subdir_path):
                continue

            for language_name in os.listdir(subdir_path):
                if not any([language_name.startswith(language_prefix) for language_prefix in language_prefixes]):
                    continue

                language_path = os.path.join(subdir_path, language_name)
                subfolder_names = os.listdir(language_path)
                subfolder_paths = [os.path.join(
                    language_path, subfolder_name) for subfolder_name in subfolder_names]
                subfolder_paths = [
                    x for x in subfolder_paths if os.path.isdir(x)]
                subfolder_paths.append(language_path)

                for subfolder_path in subfolder_paths:
                    filepaths = [os.path.join(subfolder_path, x)
                                 for x in os.listdir(subfolder_path)]
                    filepaths = [x for x in filepaths if os.path.isfile(x)]
                    for filepath in filepaths:
                        with open(filepath, 'r', encoding='utf-8') as data_file:
                            data_file_text = data_file.read().split('\n')
                            ocr_strings = self._cut_string(
                                data_file_text[1][start_position:], max_string_length)
                            gs_strings = self._cut_string(
                                data_file_text[2][start_position:], max_string_length)

                            if type(ocr_strings) is list:
                                ocr_sequences.extend(ocr_strings)
                            else:
                                ocr_sequences.append(ocr_strings)

                            if type(gs_strings) is list:
                                gs_sequences.extend(gs_strings)
                            else:
                                gs_sequences.append(gs_strings)

        result = tuple(zip(*[
            (ocr_sequence, gs_sequence)
            for (ocr_sequence, gs_sequence)
            in zip(ocr_sequences, gs_sequences)
            if ocr_sequence != '' and gs_sequence != ''
        ]))

        return result

    def _process_trove_data(self):
        cache_item_keys = self._cache_service.get_item_from_cache(
            CacheOptions(
                'trove-item-keys',
                configuration_specific=False),
            callback_function=self._download_trove_files)

        title_prefix = '*$*OVERPROOF*$*'
        separator = '||@@||'

        ocr_sequences = []
        gs_sequences = []

        for cache_item_key in cache_item_keys:
            # Get the downloaded file from the cache, process it and add it to the total collection of items
            file_content: str = self._cache_service.get_item_from_cache(
                CacheOptions(
                    cache_item_key,
                    configuration_specific=False)
            ).decode('utf-8')

            file_content_lines = file_content.splitlines()
            for file_line in file_content_lines:
                if file_line.startswith(title_prefix) or file_line == separator:
                    continue

                text_strings = file_line.split(separator)
                text_strings = self._string_process_service.convert_strings_unicode_symbols(
                    text_strings)
                text_strings = self._string_process_service.remove_strings_characters(
                    text_strings, characters=['#', '@', '\n'])

                ocr_sequences.append(text_strings[0])
                gs_sequences.append(text_strings[1])

        result = tuple(zip(*[
            (ocr_sequence, gs_sequence)
            for (ocr_sequence, gs_sequence)
            in zip(ocr_sequences, gs_sequences)
            if ocr_sequence != '' and gs_sequence != ''
        ]))

        return result

    def _download_trove_files(self):
        cache_item_keys = []

        # Download and cache all files from dataset #1
        dataset1_file_urls = [
            f'http://overproof.projectcomputing.com/datasets/dataset1/rawTextAndHumanCorrectionPairs/smh{i}.txt' for i in range(1842, 1955)]

        for i, file_url in enumerate(dataset1_file_urls):
            cache_key = f'trove-d1-{i}'
            cached_successfully = self._cache_service.download_and_cache(
                file_url,
                CacheOptions(
                    cache_key,
                    configuration_specific=False),
                overwrite=False)

            if cached_successfully:
                cache_item_keys.append(cache_key)

        # Download and cache dataset #2
        dataset2_file_url = 'http://overproof.projectcomputing.com/datasets/dataset2/rawTextAndHumanCorrectionAndOverproofCorrectionTriples/allArticles.txt'
        dataset2_key = 'trove-d2'
        cached_successfully = self._cache_service.download_and_cache(
            dataset2_file_url,
            CacheOptions(
                dataset2_key,
                configuration_specific=False),
            overwrite=False)

        if cached_successfully:
            cache_item_keys.append(dataset2_key)

        # Download and cache dataset #3
        dataset3_file_url = 'http://overproof.projectcomputing.com/datasets/dataset3/rawTextAndHumanCorrectionAndOverproofCorrectionTriples/allArticles.txt'
        dataset3_key = 'trove-d3'
        cached_successfully = self._cache_service.download_and_cache(
            dataset3_file_url,
            CacheOptions(
                dataset3_key,
                configuration_specific=False),
            overwrite=False)

        if cached_successfully:
            cache_item_keys.append(dataset3_key)

        return cache_item_keys

    def _get_folder_language_prefixes(self, language: Language) -> List[str]:
        if language == Language.English:
            return ['eng', 'EN']
        elif language == Language.French:
            return ['fr', 'FR']
        elif language == Language.German:
            return ['DE']
        elif language == Language.Dutch:
            return ['NL']
        else:
            raise NotImplementedError()
