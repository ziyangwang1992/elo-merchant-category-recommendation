import datetime


def string_to_time(string, form="%Y-%m"):
    if string is None or len(str(string)) != 7:
        string = "2015-05"
    return datetime.datetime.strptime(string, form)


def get_interval_days(str1, str2, form="%Y-%m"):
    time1 = string_to_time(str1, form)
    time2 = string_to_time(str2, form=form)
    interval = time2 - time1
    return interval.days
