# geoclip_embedding.py

import logging
from typing import Optional, Union, cast, List

import numpy as np
import torch
from geoclip import LocationEncoder

from chromadb.api.types import (
    Document,
    Documents,
    Embedding,
    EmbeddingFunction,
    Embeddings,
    is_document,
)

# Initialize logger
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# GeoClipEmbeddingFunction Module
# 
# Original GeoCLIP Reference:
# @inproceedings{geoclip,
#   title={GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization},
#   author={Vivanco, Vicente and Nayak, Gaurav Kumar and Shah, Mubarak},
#   booktitle={Advances in Neural Information Processing Systems},
#   year={2023}
# }
#
# Custom Embedding Implementation by Andrew Herr (LatticeWorks)
# -------------------------------------------------------------------------

class GeoClipEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    GeoClip Embedding Function

    This class implements an embedding function for geographic coordinates using the GeoCLIP model.
    It aligns geographic locations with image embeddings to enable effective geo-localization
    across worldwide locations.

    Original Reference:
    Vivanco, Vicente, Gaurav Kumar Nayak, and Mubarak Shah.
    "GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization."
    Advances in Neural Information Processing Systems (2023).

    Credits:
    Custom embedding implementation by Andrew Herr (LatticeWorks).

    Attributes:
        _gps_encoder (LocationEncoder): The GeoCLIP location encoder model.
        _torch (module): The PyTorch module.
        _device (str): The computation device ('cpu' or 'cuda').

    Methods:
        __init__(device: Optional[str] = None) -> None:
            Initializes the embedding function with the specified device.

        _encode_coordinates(coordinates: Union[Document, List[float]]) -> Embedding:
            Encodes a single coordinate pair into a 512-dimensional embedding.

        __call__(input: Documents) -> Embeddings:
            Processes a list of documents and generates corresponding embeddings.
    """

    # Class-level attributes
    _LocationEncoder = LocationEncoder  # GeoCLIP Location Encoder
    _torch = torch  # PyTorch module
    _device = None  # Computation device

    def __init__(self, device: Optional[str] = None) -> None:
        """
        Initializes the GeoClipEmbeddingFunction.

        Loads the GeoCLIP location encoder model and configures the computation device.

        Args:
            device (Optional[str]): The device to use for computation ('cpu' or 'cuda').
                                     If not specified, auto-detection is performed.

        Raises:
            ValueError: If the torch package is not installed.
            RuntimeError: If the model fails to move to the specified device.
        """
        # Determine the computation device
        if not device:
            self._device = "cuda" if self._torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        try:
            # Initialize the GeoCLIP location encoder
            self._gps_encoder = self._LocationEncoder()
            # Move the model to the specified device
            self._gps_encoder.to(self._device)
            # Test device availability by creating a dummy tensor
            self._torch.tensor([0.0], device=self._device)
            logger.info(f"GeoClipEmbeddingFunction initialized on device: {self._device}")
        except RuntimeError as e:
            # Handle device configuration errors by falling back to CPU
            logger.warning(
                f"Failed to move GeoCLIP model to device {self._device}: {e}. Falling back to CPU."
            )
            self._device = "cpu"
            self._gps_encoder.to(self._device)
            logger.info(f"GeoClipEmbeddingFunction initialized on device: {self._device}")

    def _encode_coordinates(self, coordinates: Union[Document, List[float]]) -> Embedding:
        """
        Encodes a single coordinate pair using the GeoCLIP model.

        Args:
            coordinates (Union[Document, List[float]]): A single coordinate pair, either as a
                                                       "lat,lon" string or a [lat, lon] list.

        Returns:
            Embedding: A 512-dimensional numpy array representing the encoded coordinates.
                      Returns a zero vector if the input is invalid.

        Raises:
            ValueError: If the coordinate format is invalid or out of range.
        """
        try:
            # Parse the input coordinates
            if isinstance(coordinates, str):
                lat, lon = map(float, coordinates.strip().split(','))
            elif isinstance(coordinates, list) and len(coordinates) == 2:
                lat, lon = coordinates
            else:
                raise ValueError(
                    "Invalid coordinate format. Expected 'lat,lon' string or [lat, lon] list."
                )

            # Validate latitude and longitude ranges
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError("Latitude must be between -90 and 90 and longitude between -180 and 180.")

            # Create a tensor for the coordinates
            gps_data = self._torch.tensor([[lat, lon]], dtype=self._torch.float32, device=self._device)
            # Generate the embedding without tracking gradients
            with self._torch.no_grad():
                gps_embedding = self._gps_encoder(gps_data).squeeze().cpu().numpy()
            return cast(Embedding, gps_embedding)
        except ValueError as e:
            # Log a warning and return a zero vector for invalid inputs
            logger.warning(f"Invalid coordinates '{coordinates}': {e}")
            return cast(Embedding, np.zeros(512))

    def __call__(self, input: Documents) -> Embeddings:
        """
        Processes a list of documents and generates corresponding embeddings.

        Args:
            input (Documents): A list of documents, where each document is either a
                               "lat,lon" string or a [lat, lon] list.

        Returns:
            Embeddings: A list of 512-dimensional numpy arrays representing the embeddings
                       of the input coordinates. Invalid inputs are represented by zero vectors.
        """
        embeddings: Embeddings = []
        for item in input:
            if is_document(item) or (isinstance(item, list) and len(item) == 2):
                # Encode valid coordinates
                embedding = self._encode_coordinates(item)
                embeddings.append(embedding)
            else:
                # Log a warning and append a zero vector for invalid inputs
                logger.warning(
                    f"Skipping invalid input: {item}. Expected 'lat,lon' string or [lat, lon] list."
                )
                embeddings.append(cast(Embedding, np.zeros(512)))
        return embeddings
