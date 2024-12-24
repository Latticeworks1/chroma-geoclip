import logging
from typing import Optional, Union, cast, List
import numpy as np

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

    This function takes a list of documents containing latitude and longitude pairs and
    generates corresponding embeddings using the GeoCLIP model. It handles various input formats
    and invalid inputs gracefully.
    """

    _LocationEncoder = None  # Class-level attribute for LocationEncoder
    _torch = None  # Class-level attribute for torch
    _device = None

    def __init__(self, device: Optional[str] = None) -> None:
        """
        Initializes the GeoClipEmbeddingFunction.

        Loads the GeoCLIP location encoder model and torch.

        Args:
            device: The device to use for computation ('cpu' or 'cuda'). Defaults to auto-detection.

        Raises:
            ValueError: If the geoclip or torch package is not installed.
        """
        if self._LocationEncoder is None:  # Only import if not already imported
            try:
                from geoclip import LocationEncoder
                self._LocationEncoder = LocationEncoder
            except ImportError:
                raise ValueError(
                    "The geoclip python package is not installed. Please install it with `pip install geoclip`."
                )
        if self._torch is None:
            try:
                import torch
                self._torch = torch
            except ImportError:
                raise ValueError(
                    "The torch python package is not installed. Please install it with `pip install torch`"
                )

        self._gps_encoder = self._LocationEncoder()

        self._device = device or ("cuda" if self._torch.cuda.is_available() else "cpu")
        try:
            self._gps_encoder.to(self._device)
            self._torch.tensor([0.0], device=self._device)  # Test device availability
        except RuntimeError as e:
            logger.warning(f"Failed to move model to device {self._device}: {e}. Falling back to CPU.")
            self._device = "cpu"
            self._gps_encoder.to(self._device)

        logger.info(f"Using device: {self._device}")

    def _encode_coordinates(self, coordinates: Union[Document, List[float]]) -> Embedding:
        """
        Encodes a single coordinate pair using the GeoCLIP model.

        Args:
            coordinates: Either a string "lat,lon" or a list [lat, lon].

        Returns:
            The encoded coordinate embedding as a numpy array, or a zero vector on error.
        """
        try:
            if isinstance(coordinates, str):
                lat, lon = map(float, coordinates.strip().split(','))
            elif isinstance(coordinates, list) and len(coordinates) == 2:
                lat, lon = coordinates
            else:
                raise ValueError("Invalid coordinate format. Expected 'lat,lon' string or [lat, lon] list")

            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError("Latitude and longitude out of range")

            gps_data = self._torch.tensor([[lat, lon]], dtype=self._torch.float32, device=self._device)
            with self._torch.no_grad():
                gps_embedding = self._gps_encoder(gps_data).squeeze().cpu().numpy()
            return cast(Embedding, gps_embedding)
        except ValueError as e:
            logger.warning(f"Could not parse coordinates: '{coordinates}'. Error: {e}")
            return cast(Embedding, np.zeros(512)) #return zero vector of correct size

    def __call__(self, input: Documents) -> Embeddings:
        """
        Processes a list of documents and generates embeddings.

        Args:
            input: A list of documents, where each document is a "lat,lon" string or a [lat, lon] list.

        Returns:
            A list of embeddings corresponding to the input documents.
        """

        embeddings: Embeddings = []
        for item in input:
            if is_document(item) or (isinstance(item, list) and len(item) == 2):
                embeddings.append(self._encode_coordinates(item))
            else:
                logger.warning(f"Skipping invalid input: {item}. Expected 'lat,lon' string or [lat, lon] list.")
                embeddings.append(cast(Embedding, np.zeros(512))) #add zero vector for invalid input

        return embeddings
