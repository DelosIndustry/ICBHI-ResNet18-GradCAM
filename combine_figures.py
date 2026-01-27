import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import glob

# ==========================================
# 1. 설정
# ==========================================
CONFIG = {
    'target_folder': './result_images',  # 이미지가 저장된 폴더
    'save_path': './result_images/Figure_Combined_Result.png', # 결과 저장 경로
    'top_k': 2  # Best와 Worst 각각 몇 개씩 보여줄지 (2x2 그리드를 위해 2 추천)
}

# ==========================================
# 2. 메인 실행
# ==========================================
if __name__ == "__main__":
    print("--- 🖼️ 이미지 합치기 작업 시작 ---")

    # 1. 파일 리스트 가져오기
    # 파일명 패턴: Best_Conf0.xxxx..._GradCAM.png
    best_files = sorted(glob.glob(os.path.join(CONFIG['target_folder'], 'Best_*_GradCAM.png')), reverse=True)
    worst_files = sorted(glob.glob(os.path.join(CONFIG['target_folder'], 'Worst_*_GradCAM.png'))) # 낮은 점수부터

    # 파일이 부족한 경우 처리
    if len(best_files) < CONFIG['top_k'] or len(worst_files) < CONFIG['top_k']:
        print("⚠️ 경고: 저장된 이미지가 충분하지 않습니다. visualize_best_worst.py를 먼저 실행했는지 확인하세요.")
        # 있는 만큼만이라도 그리도록 설정
        n_best = min(len(best_files), CONFIG['top_k'])
        n_worst = min(len(worst_files), CONFIG['top_k'])
    else:
        n_best = CONFIG['top_k']
        n_worst = CONFIG['top_k']

    # 2. Figure 생성 (2행 x top_k열)
    # 이미지 비율에 따라 figsize 조절 (가로로 긴 이미지들이므로 넓게 설정)
    fig, axes = plt.subplots(nrows=2, ncols=CONFIG['top_k'], figsize=(18, 10))
    
    # 타이틀 간격 및 전체 레이아웃 조정
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    fig.suptitle('Qualitative Analysis: Comparison of Best vs. Worst Predictions', fontsize=20, fontweight='bold', y=0.95)

    # 3. Best Cases 그리기 (윗줄)
    for i in range(n_best):
        img_path = best_files[i]
        img = mpimg.imread(img_path)
        
        # Grid가 1열일 경우 axes가 1차원 배열이 아닐 수 있으므로 처리
        ax = axes[0, i] if CONFIG['top_k'] > 1 else axes[0]
        
        ax.imshow(img)
        ax.axis('off') # 축 제거
        
        # 파일명에서 점수 추출해서 제목 달기 (옵션)
        filename = os.path.basename(img_path)
        conf_score = filename.split('_')[1].replace('Conf', '')
        ax.set_title(f"[Success] Best Case #{i+1}\n(Confidence: {float(conf_score)*100:.1f}%)", 
                     fontsize=14, color='green', fontweight='bold')

    # 4. Worst Cases 그리기 (아랫줄)
    for i in range(n_worst):
        img_path = worst_files[i]
        img = mpimg.imread(img_path)
        
        ax = axes[1, i] if CONFIG['top_k'] > 1 else axes[1]
        
        ax.imshow(img)
        ax.axis('off')
        
        filename = os.path.basename(img_path)
        conf_score = filename.split('_')[1].replace('Conf', '')
        ax.set_title(f"[Failure/Edge] Worst Case #{i+1}\n(Confidence: {float(conf_score)*100:.1f}%)", 
                     fontsize=14, color='red', fontweight='bold')

    # 5. 저장
    plt.savefig(CONFIG['save_path'], dpi=300, bbox_inches='tight')
    print(f"\n✅ 통합 이미지가 저장되었습니다: {CONFIG['save_path']}")
    print("이제 이 파일을 논문에 넣으시면 됩니다!")
    plt.show()