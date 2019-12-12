class _pa16j:
    """Pose alternated with 16 joints (like Penn Action with three more
    joints on the spine.
    """

    num_joints = 16
    joint_names = [
        "pelvis",
        "thorax",
        "neck",
        "head",
        "r_shoul",
        "l_shoul",
        "r_elb",
        "l_elb",
        "r_wrist",
        "l_wrist",
        "r_hip",
        "l_hip",
        "r_knww",
        "l_knee",
        "r_ankle",
        "l_ankle",
    ]

    """Horizontal flip mapping"""
    map_hflip = [0, 1, 2, 3, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]

    """Projections from other layouts to the PA16J standard"""
    map_from_mpii = [6, 7, 8, 9, 12, 13, 11, 14, 10, 15, 2, 3, 1, 4, 0, 5]
    map_from_ntu = [0, 20, 2, 3, 4, 8, 5, 9, 6, 10, 12, 16, 13, 17, 14, 18]

    """Projections of PA16J to other formats"""
    map_to_pa13j = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    map_to_jhmdb = [2, 1, 3, 4, 5, 10, 11, 6, 7, 12, 13, 8, 9, 14, 15]
    map_to_mpii = [14, 12, 10, 11, 13, 15, 0, 1, 2, 3, 8, 6, 4, 5, 7, 9]
    map_to_lsp = [14, 12, 10, 11, 13, 15, 8, 6, 4, 5, 7, 9, 2, 3]

    """Color map"""
    color = ["g", "r", "b", "y", "m"]
    cmap = [0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4]
    links = [
        [0, 1],
        [1, 2],
        [2, 3],
        [4, 6],
        [6, 8],
        [5, 7],
        [7, 9],
        [10, 12],
        [12, 14],
        [11, 13],
        [13, 15],
    ]


class pa16j2d(_pa16j):
    dim = 2
