def get_benchmarkModel(input_channels, output_dim):
    return   [
    [
        #Conv(input_channels,output_channels,kernel_size,padding), Maxpool(Kernel_size,Stride)
        [input_channels, 256, 3, 1], [2, 2],
        [256, 128, 3, 1], [2, 2],
        [128, 64, 3, 1], [2, 2],
    ],
    [
        # MLP layers
        [4900, 4096],
        [4096, 1024],
        [1024, output_dim]
    ]
]



def get_lenet(input_channels, output_dim):
    return   [
    [
        #Conv(input_channels,output_channels,kernel_size,padding), Maxpool(Kernel_size,Stride)
        [input_channels, 50, 3, 1], [2, 2],
        [50, 100, 3, 1], [2, 2]
    ],
    [
        # MLP layers
        [4900, 100],
        [100, output_dim]
    ]
]

def get_alexnet(input_channels, output_dim):
    return [
        [
            [input_channels, 96, 11, 2], [3, 2],       # Conv1 + MaxPool
            [96, 256, 5, 2], [3, 2],                   # Conv2 + MaxPool
            [256, 384, 3, 1],                          # Conv3
            [384, 384, 3, 1],                          # Conv4
            [384, 256, 3, 1], [3, 2]                   # Conv5 + MaxPool
        ],
        [
            [256 * 6 * 6, 4096],
            [4096, 4096],
            [4096, output_dim]
        ]
    ]


def get_vgg11(input_channels, output_dim):
    return [
        [
            [input_channels, 64, 3, 1], [2, 2],
            [64, 128, 3, 1], [2, 2],
            [128, 256, 3, 1], [2, 2],
            [256, 512, 3, 1], [2, 2],
            [512, 512, 3, 1], [2, 2]
        ],
        [[512 * 7 * 7, 4096], [4096, 4096], [4096, output_dim]]
    ]
