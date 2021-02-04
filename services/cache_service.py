from entities.cache.cache_options import CacheOptions
from enums.configuration import Configuration
import os
from services.log_service import LogService
import time
from datetime import datetime

from typing import Any, Callable
import urllib.request

from entities.timespan import Timespan

from services.arguments.arguments_service_base import ArgumentsServiceBase
from services.file_service import FileService
from services.data_service import DataService


class CacheService:
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            file_service: FileService,
            data_service: DataService,
            log_service: LogService):

        self._arguments_service = arguments_service
        self._file_service = file_service
        self._data_service = data_service
        self._log_service = log_service

        self._global_cache_folder = self._file_service.combine_path(
            self._arguments_service.cache_folder,
            create_if_missing=True)

        self._challenge_cache_folder = self._file_service.combine_path(
            self._global_cache_folder,
            self._arguments_service.challenge.value.lower(),
            create_if_missing=True)

        self._language_cache_folder = self._file_service.combine_path(
            self._challenge_cache_folder,
            self._arguments_service.language.value.lower(),
            create_if_missing=True)

        self._internal_cache_folder = self._file_service.combine_path(
            self._language_cache_folder,
            self._arguments_service.configuration.value.lower(),
            create_if_missing=True)

        self._seed_cache_folder = self._file_service.combine_path(
            self._internal_cache_folder,
            str(self._arguments_service.seed),
            create_if_missing=True)

    def get_item_from_cache(
            self,
            cache_options: CacheOptions,
            callback_function: Callable = None) -> Any:
        cached_object = None
        if self.item_exists(cache_options):
            cache_folder = self._get_cache_folder_path(cache_options)

            # try to get the cached object
            cached_object = self._data_service.load_python_obj(
                cache_folder,
                cache_options.get_item_key())

        if cached_object is None:
            # if the cached object does not exist we call the callback function to calculate it
            # and then cache it to the file system
            if callback_function is None:
                self._log_service.log_debug(
                    'Cached object was not found or was expired and no callback function was provided')
                return None

            self._log_service.log_debug(
                'Cached object was not found or was expired. Executing callback function')
            cached_object = callback_function()
            self.cache_item(cached_object, cache_options)

        return cached_object

    def load_file_from_cache(self, cache_options) -> object:
        cache_folder = self._get_cache_folder_path(cache_options)

        filepath = os.path.join(cache_folder, cache_options.get_item_key())
        with open(filepath, 'rb') as cached_file:
            result = cached_file.read()
            return result

    def cache_item(self, item: object, cache_options: CacheOptions, overwrite: bool = True):
        self._log_service.log_debug(
            f'Attempting to cached object item with key {cache_options.get_item_key()} [config-specific: {cache_options.configuration_specific} | challenge-specific: {cache_options.challenge_specific}]')
        if not overwrite and self.item_exists(cache_options):
            return

        cache_folder = self._get_cache_folder_path(cache_options)
        saved = self._data_service.save_python_obj(
            item,
            cache_folder,
            cache_options.get_item_key())

        if saved:
            self._log_service.log_debug('Object cached successfully')
        else:
            self._log_service.log_debug('Object was not cached successfully')

    def item_exists(self, cache_options: CacheOptions) -> bool:
        cache_folder = self._get_cache_folder_path(cache_options)

        result = self._data_service.check_python_object(
            cache_folder,
            cache_options.get_item_key())

        return result

    def download_and_cache(
            self,
            download_url: str,
            cache_options: CacheOptions,
            overwrite: bool = True,) -> bool:
        if not overwrite and self.item_exists(cache_options):
            return True

        cache_folder = self._get_cache_folder_path(cache_options)

        try:
            download_file_path = os.path.join(cache_folder, cache_options.get_item_key())
            self._log_service.log_debug(
                f'Attempting to download item from \'{download_url}\' to \'{download_file_path}\'')

            urllib.request.urlretrieve(
                download_url,
                download_file_path)
        except:
            self._log_service.log_error(
                f'There was error downloading file from url \'{download_url}\'')
            return False

        self._log_service.log_debug(
            f'Object was downloaded and saved successfully')
        return True

    def _get_cache_folder_path(self, cache_options: CacheOptions):
        if not cache_options.challenge_specific:
            return self._global_cache_folder

        if not cache_options.language_specific:
            return self._challenge_cache_folder

        if not cache_options.configuration_specific:
            return self._language_cache_folder

        if cache_options.configuration is not None:
            result_path = self._file_service.combine_path(
                self._language_cache_folder,
                cache_options.configuration.value.lower(),
                create_if_missing=True)

            if cache_options.seed_specific or cache_options.seed is not None:
                seed = str(self._arguments_service.seed)
                if cache_options.seed is not None:
                    seed = str(cache_options.seed)

                result_path = self._file_service.combine_path(
                    result_path,
                    seed,
                    create_if_missing=True)
        else:
            if cache_options.seed_specific:
                result_path = self._seed_cache_folder
            else:
                result_path = self._internal_cache_folder

        return result_path