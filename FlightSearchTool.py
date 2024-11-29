from langchain_core.tools import BaseTool
from pydantic.v1 import BaseModel, Field
import os
from dotenv import load_dotenv
from typing import Type, Optional
import requests

load_dotenv()
api_key = os.getenv("SKYSCANNER_KEY")


class LocationInfo(BaseModel):
    lat: str = Field(description="lattitude of the location")
    lng: str = Field(description="longitude of the location")


class AirportInfo(BaseModel):
    skyId: str = Field(description="airport code according to skyscanner")
    entityId: str = Field(
        description="entity id to uniquely identify an airport")


class FlightSearchInput(BaseModel):
    source: AirportInfo = Field(description="details of source airport")
    destination: AirportInfo = Field(
        description="details of destination airport")
    startDate: str = Field(description="start date to search for flights")
    returnDate: Optional[str] = Field(
        description="return date to search for flights")


class AirportFindTool(BaseTool):
    name: str = "airport_finder"
    description: str = "to find airport nearest to the given location with their skyId and entityId. This tool is helpful to find airport info to search for flights"
    args_schema: Type[BaseModel] = LocationInfo

    def _run(self, lat: str, lng: str) -> AirportInfo:
        airport_info = get_airport_details(lat=lat, long=lng)
        return airport_info


class FlightSearchTool(BaseTool):
    name: str = "flight_live_search"
    description: str = "to find flight schedules from source to destination airpots using skyscanner api. Requires source and destination airport details, a start date, and optionally a return date. If required inputs like the start date are missing, the agent must explicitly request them from the user."
    args_schema: Type[BaseModel] = FlightSearchInput

    def _run(self, source: AirportInfo, destination: AirportInfo, startDate: str, returnDate: Optional[str] = None):
        flight_schedule = get_flight_details(
            source, destination, startDate=startDate, returnDate=returnDate)
        return flight_schedule


def get_airport_details(lat, long):
    url = "https://sky-scrapper.p.rapidapi.com/api/v1/flights/getNearByAirports"
    querystring = {"lat": lat, "lng": long, "locale": "en-US"}
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "sky-scrapper.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)

    airport_data = response.json().get("data")["current"]
    return airport_data


def get_flight_details(source: AirportInfo, destination: AirportInfo, startDate: str, returnDate: Optional[str] = None):
    url = "https://sky-scrapper.p.rapidapi.com/api/v2/flights/searchFlights"
    querystring = {
        "originSkyId": source.skyId,
        "destinationSkyId": destination.skyId,
        "originEntityId": source.entityId,
        "destinationEntityId": destination.entityId,
        "date": startDate, "cabinClass": "economy", "adults": "1", "sortBy": "best", "currency": "INR", "market": "en-US", "countryCode": "IN"}

    if (returnDate != None):
        querystring["returnDate"] = returnDate
    # print("QueryString Inside fn:", querystring)
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "sky-scrapper.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    iteneraries = response.json()["data"]["itineraries"]
    if (len(iteneraries) > 0):
        return iteneraries
    else:
        return "No Flights founds on this date"


def structure_flight_schedules(iteneraries):
    trips = []
    obj = {}
    for trip in iteneraries:
        obj["price"] = trip["price"]["formatted"]

        for item in trip["legs"]:
            obj["souce_airport_code"] = item["origin"]["id"]
            obj["destination_airport_code"] = item["destination"]["id"]
            obj["departure"] = item["departure"]
            obj["arrival"] = item["arrival"]
            obj["duration_in_min"] = item["durationInMinutes"]
            obj["connection_flight"] = "true" if len(
                item["carriers"]["marketing"]) > 1 else "false"

        obj["score"] = trip["score"]
        trips.append(obj)
    return trips
