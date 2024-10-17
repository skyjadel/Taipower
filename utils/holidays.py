import datetime

# 列出台灣自2023年以來的國定假日以及勞動節的日期，通過觀察得知這與總發電量與台電的太陽能購電量息息相關。
holidays = [
    datetime.datetime(2023, 1, 1),
    datetime.datetime(2023, 1, 2),
    datetime.datetime(2023, 1, 20),
    datetime.datetime(2023, 1, 23),
    datetime.datetime(2023, 1, 24),
    datetime.datetime(2023, 1, 25),
    datetime.datetime(2023, 1, 26),
    datetime.datetime(2023, 1, 27),
    datetime.datetime(2023, 2, 27),
    datetime.datetime(2023, 2, 28),
    datetime.datetime(2023, 4, 3),
    datetime.datetime(2023, 4, 4),
    datetime.datetime(2023, 4, 5),
    datetime.datetime(2023, 5, 1),
    datetime.datetime(2023, 6, 22),
    datetime.datetime(2023, 6, 23),
    datetime.datetime(2023, 9, 29),
    datetime.datetime(2023, 10, 9),
    datetime.datetime(2023, 10, 10),
    datetime.datetime(2024, 1, 1),
    datetime.datetime(2024, 2, 8),
    datetime.datetime(2024, 2, 9),
    datetime.datetime(2024, 2, 12),
    datetime.datetime(2024, 2, 13),
    datetime.datetime(2024, 2, 14),
    datetime.datetime(2024, 2, 28),
    datetime.datetime(2024, 4, 4),
    datetime.datetime(2024, 4, 5),
    datetime.datetime(2024, 5, 1),
    datetime.datetime(2024, 6, 10),
    datetime.datetime(2024, 7, 25),
    datetime.datetime(2024, 9, 17),
    datetime.datetime(2024, 10, 10),
    datetime.datetime(2025, 1, 1),
]

typhoon_leave = {
    datetime.datetime(2024, 7, 24): 1,
    datetime.datetime(2024, 7, 25): 1,
    datetime.datetime(2024, 10, 1): 0.2,
    datetime.datetime(2024, 10, 2): 1,
    datetime.datetime(2024, 10, 3): 1,
    datetime.datetime(2024, 10, 4): 0.2,
}

#補班日
adjusted_work_days = [
    datetime.datetime(2023, 1, 7),
    datetime.datetime(2023, 2, 4),
    datetime.datetime(2023, 2, 18),
    datetime.datetime(2023, 3, 25),
    datetime.datetime(2023, 6, 17),
    datetime.datetime(2023, 9, 23),
    datetime.datetime(2024, 2, 3),
]

#春節
lunar_new_year = [
    datetime.datetime(2023, 1, 20),
    datetime.datetime(2023, 1, 21),
    datetime.datetime(2023, 1, 22),
    datetime.datetime(2023, 1, 23),
    datetime.datetime(2023, 1, 24),
    datetime.datetime(2023, 1, 25),
    datetime.datetime(2023, 1, 26),
    datetime.datetime(2023, 1, 27),
    datetime.datetime(2023, 1, 28),
    datetime.datetime(2023, 1, 29),
    datetime.datetime(2024, 2, 8),
    datetime.datetime(2024, 2, 9),
    datetime.datetime(2024, 2, 10),
    datetime.datetime(2024, 2, 11),
    datetime.datetime(2024, 2, 12),
    datetime.datetime(2024, 2, 13),
    datetime.datetime(2024, 2, 14),   
]