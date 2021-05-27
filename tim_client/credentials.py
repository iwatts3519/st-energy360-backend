from typing import Dict, List, Tuple
from base64 import b64encode
import logging
from logging import Logger
from tim_client._version import __version__, __client_name__, __min_engine_version__


class Credentials:
    """TIM License data necessary to use TIM Engine."""

    __logger: Logger = None
    __tim_url: str = None
    __access_token: str = None
    __api_key: str = None
    __license_key: str = None
    __email: str = None
    __password: str = None

    def __init__(self, license_key: str = None, email: str = None, password: str = None, tim_url: str = 'https://timws.tangent.works/v4/api', api_key: str = 'a37b7c81-5a85-4eda-847d-d2d5ba33d2c8', access_token: str = None, logger: Logger = None):
        """Set TIM License credentials."""
        if self.__logger is None:
            self.__logger = logging.getLogger(__name__)

        self.set_tim_url(tim_url)
        self.__api_key = api_key

        if access_token:
            self.set_token(access_token)
        self.__license_key = license_key
        self.__email = email
        self.__password = password

    @property
    def tim_url(self):
        return self.__tim_url

    @tim_url.setter
    def tim_url(self, url):
        self.set_tim_url(url)

    @property
    def access_token(self):
        return self.__access_token

    @access_token.setter
    def access_token(self, access_token):
        self.set_token(access_token)

    @property
    def license_key(self):
        return self.__license_key

    @license_key.setter
    def license_key(self, license_key):
        self.__license_key = license_key

    @property
    def email(self):
        return self.__email

    @email.setter
    def email(self, email):
        self.__email = email

    @property
    def password(self):
        if self.__password is not None:
            return '*'*8
        return None

    @password.setter
    def password(self, password):
        self.__password = password

    def set_credentials(self, license_key: str, email: str, password: str):
        self.__license_key = license_key
        self.__email = email
        self.__password = password

    def delete_credentials(self):
        self.__license_key = None
        self.__email = None
        self.__password = None

    def set_token(self, token: str):
        if token.startswith('Bearer '):
            self.__access_token = token
        else:
            self.__access_token = f'Bearer {token}'

    def delete_token(self):
        self.__access_token = None

    def get_tim_url(self) -> str:
        """Return URL of TIM Engine."""
        return self.__tim_url

    def set_tim_url(self, tim_url: str) -> str:
        """Set new URL of TIM Engine."""
        if tim_url is not None and len(tim_url) > 0:
            self.__tim_url = tim_url
            self.__logger.debug('TIM URL has been changed to %s', tim_url)
        else:
            raise ValueError('TIM URL cannot be empty')

    def __str__(self) -> str:
        if self.__access_token:
            return f'Access token {self.__access_token[0:10]}...{self.__access_token[-3:]}'
        elif self.__license_key and self.__email and self.__password:
            return f'{self.__license_key} - {self.__email}'
        return 'Credentials not initialized'

    def __calculate_basic_auth_value(self) -> str:
        """Return value of Basic Authorization HTTP header."""
        return 'Basic %s' % b64encode(f'{self.__email}:{self.__password}'.encode()).decode("ascii")

    def get_auth_headers(self) -> Dict:
        """Return dict with authoriaztion headers required by TIM Engine."""
        if self.__access_token:
            return {
                'Authorization': self.__access_token,
                'Api-Key': self.__api_key,
                'X-Client-Name': __client_name__,
                'X-Client-Version': __version__,
                'X-Min-Version': __min_engine_version__
            }
        elif self.__license_key and self.__email and self.__password:
            return {
                'Authorization': self.__calculate_basic_auth_value(),
                'License-Key': self.__license_key,
                'Api-Key': self.__api_key,
                'X-Client-Name': __client_name__,
                'X-Client-Version': __version__,
                'X-Min-Version': __min_engine_version__
            }
        else:
            raise ValueError('Credentials are not configured')
