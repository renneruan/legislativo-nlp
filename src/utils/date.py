from datetime import datetime, timedelta


def get_weekly_dates():
    today = datetime.today()
    weekday = today.weekday()

    if weekday < 5:
        start_date = today

        days_until_saturday = 5 - today.weekday()  # Sábado é valor 5
        end_date = today + timedelta(days=days_until_saturday)

    elif weekday == 5:
        start_date = today + timedelta(days=1)
        end_date = today + timedelta(days=7)

    else:
        start_date = today
        end_date = today + timedelta(days=6)

    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
