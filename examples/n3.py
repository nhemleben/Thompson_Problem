from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from search import search

E,x=search(
    3,
    target_depth=10
)

print(E,"Should be \sqrt{3} = 1.732...")
print(x)