from dateutil import rrule




def weeks_between(start_date, end_date):
    """
    This function compute weeks between 2 date

    Parameters
    ----------
    start_date: datetime
    end_date: datetime

    Returns
    -------
    weeks: int
    """
    weeks = rrule.rrule(rrule.WEEKLY, dtstart=start_date, until=end_date)
    return weeks.count() - 1
