from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from search import search

E,x=search(
    2,
    target_depth=8
)

print(E,"Should be 1/2")
print(x)