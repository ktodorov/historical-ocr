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
            '.cache', create_if_missing=True)

        self._challenge_cache_folder = self._file_service.combine_path(
            self._global_cache_folder,
            self._arguments_service.challenge.value.lower(),
            create_if_missing=True)

        self._internal_cache_folder = self._file_service.combine_path(
            self._challenge_cache_folder,
            self._arguments_service.configuration.value.lower(),
            self._arguments_service.language.value.lower(),
            create_if_missing=True)

    def get_item_from_cache(
            self,
            item_key: str,
            callback_function: Callable = None,
            time_to_keep: Timespan = None,
            configuration_specific: bool = True,
            challenge_specific: bool = True) -> Any:
        cache_folder = self._get_cache_folder_path(
            configuration_specific=configuration_specific,
            challenge_specific=challenge_specific)

        # try to get the cached object
        cached_object = self._data_service.load_python_obj(
            cache_folder,
            item_key)

        if cached_object is None or self._cache_has_expired(item_key, time_to_keep, configuration_specific, challenge_specific):

            # if the cached object does not exist or has expired we call
            # the callback function to calculate it and then cache it to the file system
            if callback_function is None:
                self._log_service.log_debug('Cached object was not found or was expired and no callback function was provided')
                return None

            self._log_service.log_debug('Cached object was not found or was expired. Executing callback function')
            cached_object = callback_function()
            self.cache_item(
                item_key, 
                cached_object, 
                configuration_specific=configuration_specific,
                challenge_specific=challenge_specific)

        return cached_object

    def load_file_from_cache(
            self,
            item_key: str,
            configuration_specific: bool = True,
            challenge_specific: bool = True) -> object:
        cache_folder = self._get_cache_folder_path(
            configuration_specific=configuration_specific,
            challenge_specific=challenge_specific)

        filepath = os.path.join(cache_folder, item_key)
        with open(filepath, 'rb') as cached_file:
            result = cached_file.read()
            return result

    def cache_item(
            self,
            item_key: str,
            item: object,
            overwrite: bool = True,
            configuration_specific: bool = True,
            challenge_specific: bool = True):
        self._log_service.log_debug(f'Attempting to cached object item with key {item_key} [config-specific: {configuration_specific} | challenge-specific: {challenge_specific}]')
        if not overwrite and self.item_exists(item_key):
            return

        cache_folder = self._get_cache_folder_path(
            configuration_specific=configuration_specific,
            challenge_specific=challenge_specific)

        saved = self._data_service.save_python_obj(
            item,
            cache_folder,
            item_key)

        if saved:
            self._log_service.log_debug('Object cached successfully')
        else:
            self._log_service.log_debug('Object was not cached successfully')


    def item_exists(
            self,
            item_key: str,
            configuration_specific: bool = True,
            challenge_specific: bool = True) -> bool:
        cache_folder = self._get_cache_folder_path(
            configuration_specific=configuration_specific,
            challenge_specific=challenge_specific)

        result = self._data_service.check_python_object(
            cache_folder,
            item_key)

        return result

    def download_and_cache(
            self,
            item_key: str,
            download_url: str,
            overwrite: bool = True,
            configuration_specific: bool = True,
            challenge_specific: bool = True) -> bool:
        if not overwrite and self.item_exists(item_key):
            return True

        cache_folder = self._get_cache_folder_path(
            configuration_specific=configuration_specific,
            challenge_specific=challenge_specific)

        try:
            download_file_path = os.path.join(cache_folder, item_key)
            self._log_service.log_debug(f'Attempting to download item from \'{download_url}\' to \'{download_file_path}\'')

            urllib.request.urlretrieve(
                download_url,
                download_file_path)
        except:
            self._log_service.log_error(f'There was error downloading file from url \'{download_url}\'')
            return False

        self._log_service.log_debug(f'Object was downloaded and saved successfully')
        return True

    def _get_cache_folder_path(
            self,
            challenge_specific: bool,
            configuration_specific: bool):
        if not challenge_specific:
            return self._global_cache_folder

        if not configuration_specific:
            return self._challenge_cache_folder

        return self._internal_cache_folder

    def _cache_has_expired(
            self,
            item_key: str,
            time_to_keep: Timespan,
            configuration_specific: bool,
            challenge_specific: bool) -> bool:
        if time_to_keep is None:
            return False

        cache_folder = self._get_cache_folder_path(
            configuration_specific=configuration_specific,
            challenge_specific=challenge_specific)

        item_path = os.path.join(cache_folder, f'{item_key}.pickle')

        if not os.path.exists(item_path):
            return True

        file_mtime = os.path.getmtime(item_path)
        file_datetime = datetime.fromtimestamp(file_mtime)
        current_datetime = datetime.now()
        datetime_diff = (file_datetime - current_datetime)

        if datetime_diff.microseconds > time_to_keep.milliseconds:
            return True

        return False
