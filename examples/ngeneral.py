from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from search import search

E,x=search(
    6,
    target_depth=22,
    visualize_search=False,
    show_progress=True,
)

print(E,"For n =6 should be 12.712062...")
print('actually 9.985281374 per wikipedia')
print(x)