"""ALS with signed, rating-sensitive target; confidence equals baseline IALS."""
from .ials import IALS

class CustomALS(IALS):
    def targets(self, values):
        return (values - 3.0) / 2.0
