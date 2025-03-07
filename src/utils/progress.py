import sys
import time

def progress(iterable, total_steps, desc="Progress"):
    start_time = time.time()

    for i, item in enumerate(iterable, start=1):
        elapsed_time = time.time() - start_time
        avg_time_per_step = elapsed_time / i
        remaining_time = avg_time_per_step * (total_steps - i)

        mins, secs = divmod(int(remaining_time), 60)
        eta_str = f"{mins:02d}m {secs:02d}s"

        percent_complete = (i / total_steps) * 100
        sys.stdout.write(f"\r{desc}: {percent_complete:.2f}% completed | ETA: {eta_str}")
        sys.stdout.flush()

        yield item

    print()  # Move to a new line after completion
