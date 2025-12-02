from scripts import *
from objects import *
from sympy import symbols, Matrix
import numpy as np


E, A, I = symbols('E A I')
# Define nodes from problem statement
a = Node(0, 3, 'pinned', ['x', 'y'])   # Node a at (0,3) pinned
b = Node(4, 0, 'free')                 # Node b at (4,0) free
c = Node(8, 0, 'free')                 # Node c at (8,0) free
d = Node(12, 3, 'pinned', ['x', 'y'])  # Node d at (12,3) pinned


# Define members from problem statement
members = [
    Member('truss', a, b, E=E, A=A, I=I),  # Member ab
    Member('truss', b, c, E=E, A=A, I=I),  # Member bc
    Member('truss', c, d, E=E, A=A, I=I),  # Member cd
]

K, note = preassemblyTrusses(members)


print("a) Assemble global stiffness matrix K and test for singularity:")
for row in np.array(K):
    print(row)
print("\nNote:")
print(note)

print("\nDetermining if K is singular... (Matrix(K).det() == 0)")
if Matrix(K).det() == 0:
    print("------------------\n  K is singular.\n------------------\n")
else:
    print("------------------\n  K is not singular.\n------------------\n")


print("""b) Can you get a useful solution when Pb and Pc are equal?

        
        No, because the structure is unstable and will undergo rigid body motion.
        However, when Pb and Pc are equal, the structure is in a state of equilibrium
        and the nodes will not move. Thus, the displacements at nodes b and c are
        zero, and the reactions at nodes a and d will be equal to the applied loads
        at nodes b and c, respectively.

        If this is considered to be a \"useful\" solution, then yes, we can get a useful
        solution when Pb and Pc are equal.
        """)

print("""c) What happens if you make Pc > Pb different?
      
        If Pc > Pb, the structure will still be unstable and will undergo rigid body motion.
        The structure will shift to the right, and you will see non-zero displacements at nodes b and c.
        """)


print("""d) What is the degree of static indeterminacy of this structure?

        # members = 3
        # reactions = 4 (2 at a, 2 at d)
        # nodes = 4, each with 2 DOF
        DSI = members + reactions - 2*nodes
        DSI = 3 + 4 - 2*4
        DSI = -1
        DSI = -1 indicates the structure is unstable.
        """)


print("""e) Would it be safe to build the structure as shown?
      
        Functionally, this is a rope bridge. Rope bridges are used in real life,
        and are safe, but they have significant sway and other limitations.
        This structure would be fine for small use cases that can handle such
        sway, but it would not be suitable for general use.
        """)

print("""f) How would the problem change if the structure was flipped upside down?
        
        If the structure was flipped upside down, it would still not be a stable structure.
        Once the point loads are unbalanced, the hinges would rotate, and the structure
        would invert to the position shown in the original problem statement.
        
        You will need some kind of restriction to rotation to allow for the structure to 
        be flipped upside down and remain stable. 
        """)
