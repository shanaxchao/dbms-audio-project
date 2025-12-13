import os
import pandas as pd

# 1) 메타 파일 읽기 (위에 저장한 파일 이름)
meta_path = "meta.tsv"   # 파일 이름 / 경로 맞게 수정
df = pd.read_csv(meta_path, sep="\t")

# 2) 남길 파일 이름 집합 만들기
keep_names = set(df["path"].astype(str))

# 3) mp3 파일들이 있는 루트 폴더 경로
ROOT_DIR = r"C:\Users\SAMSUNG\Desktop\DBMS\audio_project\dbms-audio-project\data\korean_2"  # 이 줄만 네 경로에 맞게 수정

# 4) 먼저 어떤 파일이 지워질지 확인만 하는 드라이런 모드
to_delete = []

for dirpath, dirnames, filenames in os.walk(ROOT_DIR):
    for fname in filenames:
        if fname.lower().endswith(".mp3") and fname not in keep_names:
            full_path = os.path.join(dirpath, fname)
            to_delete.append(full_path)

print("지울 mp3 개수:", len(to_delete))
for p in to_delete[:20]:  # 너무 많으면 앞에 20개만 보기
    print(p)

# ===== 진짜 삭제하려면 아래 주석 풀기 =====
for p in to_delete:
    os.remove(p)
print("삭제 완료")
