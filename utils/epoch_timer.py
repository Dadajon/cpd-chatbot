
def epoch_time(start_time, end_time):
    """Calculate runtime

    Args:
        start_time (): training start time
        end_time (): training end time

    Returns:
        int: minutes and seconds
    """    
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time/60)
    elapsed_secs = int(elapsed_time-(elapsed_mins*60))

    return elapsed_mins, elapsed_secs