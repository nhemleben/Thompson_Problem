def initial_cell(n):

    bounds=[]

    for i in range(n):

        bounds.append([
            [0,3.1415926535],
            [0,6.283185307]
        ])

    return Cell(bounds)