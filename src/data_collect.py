import pandas as pd
import requests
from datetime import datetime, timedelta

from pdfminer.high_level import extract_text
from bs4 import BeautifulSoup

from urllib.parse import urlparse, parse_qs
import os
