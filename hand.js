const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const targetMessage = document.getElementById('targetMessage'); // 목표 손 메시지
const currentMessage = document.getElementById('currentMessage'); // 현재 인식된 손 메시지
const modelURL = 'model/model.json'; // 모델 파일 경로 (폴더 구조에 주의)
const handLabels = ['오른손', '왼손', 'none'];
let isPlaying = false;
let targetHand = -1; // 목표 손의 클래스 (0: 오른손, 1: 왼손, 2: none)
let webcam; // webcam 변수를 더 넓은 범위에서 정의
let model; // 모델 변수를 더 넓은 범위에서 정의

// 웹 카메라 스트림 시작
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;

        // webcam 변수를 더 넓은 범위에서 정의
        webcam = await tf.data.webcam(video);

        // 웹 카메라 스트림을 계속 실행
        await tf.nextFrame();
    } catch (err) {
        console.error('카메라를 시작할 수 없습니다:', err);
    }
}

// 이미지를 4차원으로 변환 (배치 크기 1을 추가) 및 모델 예측 호출
async function processImage() {
    const img = await webcam.capture();

    // 이미지를 224x224 크기로 resize
    const resizedImg = tf.image.resizeBilinear(img, [224, 224]);

    // 이미지를 0~255 사이의 값으로 정규화
    const normalizedImg = resizedImg.toFloat().div(tf.scalar(255));

    // 이미지를 4차원으로 변환 (배치 크기 1을 추가)
    const inputImg = normalizedImg.expandDims(0);

    const result = await model.predict(inputImg);

    // 이후로 처리할 코드를 여기에 추가하세요.
}

// 모델 로드 및 초기화
async function loadModel() {
    try {
        model = await tf.loadLayersModel(modelURL);
        console.log('모델 로딩 성공.');
    } catch (error) {
        console.error('모델 로딩 오류:', error);
    }
}
// 손의 클래스 예측
async function predictHand() {
    let successFlag = false;
    while (isPlaying) {
        const img = await webcam.capture();
        const inputImg = tf.image.resizeBilinear(img, [224, 224]).toFloat().div(tf.scalar(255)).expandDims(0);
        const result = await model.predict(inputImg);
        const classIdx = result.argMax().dataSync()[0];
        
        // 사용자의 손 위치 실시간 표시
        currentMessage.textContent = `현재: ${handLabels[classIdx]}`;
        
        // 사용자의 손이 목표 손과 일치하는지 확인 후 반환
        if (classIdx === targetHand) {
            successFlag = true;
            break;
        }
  
        img.dispose();
        await tf.nextFrame();
    }
    return successFlag;
  }



// 게임 시작
async function startGame() {
    if (!isPlaying) {
         await loadModel(); // 모델 로드
         isPlaying = true;
  
         // 랜덤한 목표 손 클래스 선택 (0, 1, 2 중 하나)
         targetHand = Math.floor(Math.random() * 3);
  
         // 목표 손 메시지 표시
         targetMessage.textContent = `카메라에 ${handLabels[targetHand]}을 맞추세요.`;
  
         let successFlag; 
  
         setTimeout(async () => { 
             successFlag = await predictHand(); // 예측 시작 및 성공 여부 판단
  
             stopGame();
  
             if(successFlag) {
                 targetMessage.textContent += ' - 성공적으로 마무리!';
             } else {
                 targetMessage.textContent += ' - 시간 초과!';
             }
  
        }, 10000); // 10초 후 게임 종료
    }   //<-- 이 부분에 } 추가!
}   //<-- 이 부분에도 } 추가!

// 게임 종료
function stopGame() {
    isPlaying = false;
}

// 웹 페이지 로드 시 카메라 시작
window.addEventListener('load', async () => {
    await startCamera(); // await 키워드 추가
  
    // 게임 시작 버튼 클릭 시 게임 시작
    const startButton = document.getElementById('startButton');
    startButton.addEventListener('click', startGame);
});
  
// 카메라 스트림 종료 시 게임 종료
video.addEventListener('ended', stopGame);