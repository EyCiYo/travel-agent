import geocoder
from langchain.tools import BaseTool, Tool
from pydantic.v1 import BaseModel, Field
from typing import Type
import os
from dotenv import load_dotenv
from opencage.geocoder import OpenCageGeocode

load_dotenv()

api_key = os.getenv("GEOCODER_API_KEY")


class LocationInput(BaseModel):
    location: str = Field(
        description="name of the place whose location is to be found")


class LocationTool(BaseTool):
    name: str = "location_finder"
    description: str = "to get the location(lattitude and longitude) of a place or the user"
    args_schema: Type[BaseModel] = LocationInput

    def _run(self, location: str):
        try:
            if "current" in location.lower():
                current_location = geocoder.ip('me')
                if current_location.ok:
                    return {
                        "latitude": current_location.lat,
                        "longitude": current_location.lng,
                        "address": current_location.address or "Location not found",
                    }
                return "Unable to fetch current location."

            search_location = geocoder.opencage(location, key=api_key)
            if search_location.ok:
                return {
                    "latitude": search_location.lat,
                    "longitude": search_location.lng,
                    "address": search_location.address or "Address not found",
                }
            return f"Unable to fetch location for '{location}'. Please check the input."
        except Exception as e:
            return f"An error occurred while fetching location: {e}"
