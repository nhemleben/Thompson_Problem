from pathlib import Path
import sys
import os

sys.path.append(str(Path(__file__).resolve().parents[1]))

from search import search

workers = max(1, (os.cpu_count() or 1) - 1)

E,x=search(
    5,
    target_depth=22,
    visualize_search=False,
    show_progress=True,
    parallel_child_bounds=True,
    parallel_workers=workers,
    parallel_batch_size=32,
)

print(E,"For n =6 should be 12.712062...")
print('actually 9.985281374 per wikipedia')
print(x)
