import importlib
import logging
from typing import Optional, Union, cast

from chromadb.api.types import (
    Document,
    Documents,
    Embedding,
    EmbeddingFunction,
    Embeddings,
    is_document,
)

logger = logging.getLogger(__name__)


class GeoClipEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    Implements an embedding function for geographic coordinates using the GeoCLIP model.

    This function takes a list of documents containing latitude and longitude pairs separated by a comma
    and generates corresponding embeddings using the GeoCLIP model. It handles invalid inputs gracefully.
    """

    def __init__(self) -> None:
        """
        Initializes the GeoClipEmbeddingFunction.

        Loads the GeoCLIP location encoder model.

        Raises:
            ValueError: If the geoclip package is not installed.
        """
        try:
            from geoclip import LocationEncoder
            self._LocationEncoder = LocationEncoder  # Store the class to avoid repeated imports
        except ImportError:
            raise ValueError(
                "The geoclip python package is not installed. Please install it with `pip install geoclip`."
            )
        try:
            self._torch = importlib.import_module("torch")
        except ImportError:
            raise ValueError(
                "The torch python package is not installed. Please install it with `pip install torch`"
            )
        self._gps_encoder = self._LocationEncoder()

    def _encode_coordinates(self, coordinates: Document) -> Embedding:
        """
        Encodes a single coordinate pair using the GeoCLIP model.

        Args:
            coordinates: A string representing a latitude,longitude pair separated by a comma.

        Returns:
            The encoded coordinate embedding as a numpy array.
        Raises:
            ValueError: If the input string cannot be parsed into valid coordinates.
        """
        try:
            lat, lon = map(float, coordinates.strip().split(','))  # Remove leading/trailing whitespace
            # Validate latitude and longitude ranges
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError("Latitude and longitude out of range")
            gps_data = self._torch.tensor([[lat, lon]], dtype=self._torch.float32)  # Explicit dtype
            with self._torch.no_grad():
                gps_embedding = self._gps_encoder(gps_data).squeeze().cpu().numpy()
            return cast(Embedding, gps_embedding)
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not parse coordinates: '{coordinates}'. Error: {e}")
            return cast(Embedding, np.zeros(512))

    def __call__(self, input: Documents) -> Embeddings:
        """
        Processes a list of documents (latitude,longitude pairs) and generates embeddings.

        Args:
            input: A list of strings representing latitude,longitude pairs separated by a comma.

        Returns:
            A list of embeddings corresponding to the input documents.
        """

        embeddings: Embeddings = []
        for item in input:
            if is_document(item):
                embeddings.append(self._encode_coordinates(cast(Document, item)))
        return embeddings
