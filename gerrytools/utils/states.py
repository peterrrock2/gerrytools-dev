from dataclasses import dataclass


@dataclass(frozen=True)
class State:
    """
    A dataclass to represent a U.S. state.

    Attributes:
        abbr (str): The two-letter abbreviation of the state. Defaults to "N/A".
        name (str): The full name of the state. Defaults to "N/A".
        fips (str): The Federal Information Processing Standards (FIPS) code of the state. Defaults to "99".
    """

    abbr: str = "N/A"
    name: str = "N/A"
    fips: str = "99"


@dataclass(frozen=True)
class AL(State):
    abbr: str = "AL"
    name: str = "Alabama"
    fips: str = "01"


@dataclass(frozen=True)
class AK(State):
    abbr: str = "AK"
    name: str = "Alaska"
    fips: str = "02"


@dataclass(frozen=True)
class AZ(State):
    abbr: str = "AZ"
    name: str = "Arizona"
    fips: str = "04"


@dataclass(frozen=True)
class AR(State):
    abbr: str = "AR"
    name: str = "Arkansas"
    fips: str = "05"


@dataclass(frozen=True)
class CA(State):
    abbr: str = "CA"
    name: str = "California"
    fips: str = "06"


@dataclass(frozen=True)
class CO(State):
    abbr: str = "CO"
    name: str = "Colorado"
    fips: str = "08"


@dataclass(frozen=True)
class CT(State):
    abbr: str = "CT"
    name: str = "Connecticut"
    fips: str = "09"


@dataclass(frozen=True)
class DE(State):
    abbr: str = "DE"
    name: str = "Delaware"
    fips: str = "10"


@dataclass(frozen=True)
class FL(State):
    abbr: str = "FL"
    name: str = "Florida"
    fips: str = "12"


@dataclass(frozen=True)
class GA(State):
    abbr: str = "GA"
    name: str = "Georgia"
    fips: str = "13"


@dataclass(frozen=True)
class HI(State):
    abbr: str = "HI"
    name: str = "Hawaii"
    fips: str = "15"


@dataclass(frozen=True)
class ID(State):
    abbr: str = "ID"
    name: str = "Idaho"
    fips: str = "16"


@dataclass(frozen=True)
class IL(State):
    abbr: str = "IL"
    name: str = "Illinois"
    fips: str = "17"


@dataclass(frozen=True)
class IN(State):
    abbr: str = "IN"
    name: str = "Indiana"
    fips: str = "18"


@dataclass(frozen=True)
class IA(State):
    abbr: str = "IA"
    name: str = "Iowa"
    fips: str = "19"


@dataclass(frozen=True)
class KS(State):
    abbr: str = "KS"
    name: str = "Kansas"
    fips: str = "20"


@dataclass(frozen=True)
class KY(State):
    abbr: str = "KY"
    name: str = "Kentucky"
    fips: str = "21"


@dataclass(frozen=True)
class LA(State):
    abbr: str = "LA"
    name: str = "Louisiana"
    fips: str = "22"


@dataclass(frozen=True)
class ME(State):
    abbr: str = "ME"
    name: str = "Maine"
    fips: str = "23"


@dataclass(frozen=True)
class MD(State):
    abbr: str = "MD"
    name: str = "Maryland"
    fips: str = "24"


@dataclass(frozen=True)
class MA(State):
    abbr: str = "MA"
    name: str = "Massachusetts"
    fips: str = "25"


@dataclass(frozen=True)
class MI(State):
    abbr: str = "MI"
    name: str = "Michigan"
    fips: str = "26"


@dataclass(frozen=True)
class MN(State):
    abbr: str = "MN"
    name: str = "Minnesota"
    fips: str = "27"


@dataclass(frozen=True)
class MS(State):
    abbr: str = "MS"
    name: str = "Mississippi"
    fips: str = "28"


@dataclass(frozen=True)
class MO(State):
    abbr: str = "MO"
    name: str = "Missouri"
    fips: str = "29"


@dataclass(frozen=True)
class MT(State):
    abbr: str = "MT"
    name: str = "Montana"
    fips: str = "30"


@dataclass(frozen=True)
class NE(State):
    abbr: str = "NE"
    name: str = "Nebraska"
    fips: str = "31"


@dataclass(frozen=True)
class NV(State):
    abbr: str = "NV"
    name: str = "Nevada"
    fips: str = "32"


@dataclass(frozen=True)
class NH(State):
    abbr: str = "NH"
    name: str = "New Hampshire"
    fips: str = "33"


@dataclass(frozen=True)
class NJ(State):
    abbr: str = "NJ"
    name: str = "New Jersey"
    fips: str = "34"


@dataclass(frozen=True)
class NM(State):
    abbr: str = "NM"
    name: str = "New Mexico"
    fips: str = "35"


@dataclass(frozen=True)
class NY(State):
    abbr: str = "NY"
    name: str = "New York"
    fips: str = "36"


@dataclass(frozen=True)
class NC(State):
    abbr: str = "NC"
    name: str = "North Carolina"
    fips: str = "37"


@dataclass(frozen=True)
class ND(State):
    abbr: str = "ND"
    name: str = "North Dakota"
    fips: str = "38"


@dataclass(frozen=True)
class OH(State):
    abbr: str = "OH"
    name: str = "Ohio"
    fips: str = "39"


@dataclass(frozen=True)
class OK(State):
    abbr: str = "OK"
    name: str = "Oklahoma"
    fips: str = "40"


@dataclass(frozen=True)
class OR(State):
    abbr: str = "OR"
    name: str = "Oregon"
    fips: str = "41"


@dataclass(frozen=True)
class PA(State):
    abbr: str = "PA"
    name: str = "Pennsylvania"
    fips: str = "42"


@dataclass(frozen=True)
class RI(State):
    abbr: str = "RI"
    name: str = "Rhode Island"
    fips: str = "44"


@dataclass(frozen=True)
class SC(State):
    abbr: str = "SC"
    name: str = "South Carolina"
    fips: str = "45"


@dataclass(frozen=True)
class SD(State):
    abbr: str = "SD"
    name: str = "South Dakota"
    fips: str = "46"


@dataclass(frozen=True)
class TN(State):
    abbr: str = "TN"
    name: str = "Tennessee"
    fips: str = "47"


@dataclass(frozen=True)
class TX(State):
    abbr: str = "TX"
    name: str = "Texas"
    fips: str = "48"


@dataclass(frozen=True)
class UT(State):
    abbr: str = "UT"
    name: str = "Utah"
    fips: str = "49"


@dataclass(frozen=True)
class VT(State):
    abbr: str = "VT"
    name: str = "Vermont"
    fips: str = "50"


@dataclass(frozen=True)
class VA(State):
    abbr: str = "VA"
    name: str = "Virginia"
    fips: str = "51"


@dataclass(frozen=True)
class WA(State):
    abbr: str = "WA"
    name: str = "Washington"
    fips: str = "53"


@dataclass(frozen=True)
class WV(State):
    abbr: str = "WV"
    name: str = "West Virginia"
    fips: str = "54"


@dataclass(frozen=True)
class WI(State):
    abbr: str = "WI"
    name: str = "Wisconsin"
    fips: str = "55"


@dataclass(frozen=True)
class WY(State):
    abbr: str = "WY"
    name: str = "Wyoming"
    fips: str = "56"
