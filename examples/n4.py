from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from search import search

E,x=search(
    4,
    target_depth=13,
    visualize_search=True
)

print(E,"Should be 3 * \sqrt{3/2} = 3.674234...")
print(x)