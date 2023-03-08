from timeit import default_timer
# timer decorator
def timer(func):
    def wrapper(*args, **kwargs):
        start = default_timer()
        re = func(*args, **kwargs)
        end = default_timer()
        print("{0} time lapse: {1:.2f} s".format(func.__name__, end - start))
        return re
    return wrapper