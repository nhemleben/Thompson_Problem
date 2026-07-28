from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from search import search

E,x=search(
    5,
    target_depth=13,
    visualize_search=True,
)

print(E,"For n =5 should be 6.474691495...")
print(x)