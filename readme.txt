hierarchy 

<temp>
└──────<extri_data>
	└── images
	    ├── 1
		└── xxxxxx.jpg
	    ├── 2
		└── xxxxxx.jpg
	    ├── 3
		└── xxxxxx.jpg
	    └── 4
		└── xxxxxx.jpg
└──────<intri_data>
	└── output
		└── intri.yml


step1. chess_board 탐지
<실행> python3 detect_chessboard_ver4.py
<비고>line 202 에서 extri_data 주소 수정 
      line 206 에서 grid size 변경(AO: 0.23, A2: 0.116)
<결과> extri_data 폴더내 chessboard 생성

step2. extrinsic parameter 계산
<실행> calib_extri_jk.py
<비고>line 127 에서 extri_data 주소 수정 
      line 131 에서 intri_data/output/intri.yml 주소 수정 
<결과> intri_data/output/extri.yml 생성

step3. calibration 결과 체크
<실행> check_calib_jk.py
<비고> line 467 에서 extri_data 주소 수정
       line 472 에서 intri_data/output 주소 수정

