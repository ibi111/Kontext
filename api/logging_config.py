import os
import logging
import warnings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.ERROR,
    format="%(levelname)s - %(message)s"
)

# Suppress noisy libraries
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


try:
    from transformers import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
except ImportError:
    pass

try:
    from huggingface_hub.utils import logging as hub_logging

    hub_logging.set_verbosity_error()
except ImportError:
    pass