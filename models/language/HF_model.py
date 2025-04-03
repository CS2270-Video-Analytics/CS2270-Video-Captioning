import sys
sys.path.append('..')
from config import Config
import os
import io
import base64
from ..model import Model
import ollama
from PIL import Image
import torch

from typing import Optional, Dict
from time import time
from torchvision import transforms
import requests
import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM