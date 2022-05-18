forcalibration.sh 설명

<실행> bash forcalibration.sh
다음 값 수정하고 실행할 것
v_path
v_pattern
v_grid 


<입력 hierarchy>

<KETI_cal>
└──────<extri_data>
	└── video
	    ├── 1
		└── xxxxxx.mp4
	    └── 2
		└── xxxxxx.mp4
└──────<intri_data>
	└── output
		└── intri.yml

<출력 hierarchy>

<KETI_cal>
└──────<extri_data>
	└── video
	    ├── 1
		└── xxxxxx.mp4
	    └── 2
		└── xxxxxx.mp4
	└── images
	    ├── 1
		└── xxxxxx.jpg
	    └── 2
		└── xxxxxx.jpg
	└── chessboard
	    ├── 1
		└── xxxxxx.json
	    └── 2
		└── xxxxxx.json
└──────<intri_data>
	└── output
		└── intri.yml
		└── extri.yml
