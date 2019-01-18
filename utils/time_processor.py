import datetime


def string_to_time(str, form="%Y-%m"):
    return datetime.datetime.strptime(str, form)


def get_interval_days(str1, str2, form="%Y-%m"):
    time1 = string_to_time(str1, form)
    time2 = string_to_time(str2, form=form)
    interval = time2 - time1
    return interval.days
