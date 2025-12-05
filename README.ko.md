# Manual Backprop Neural Net: Deconstructing Deep Learning

🌍 [English](README.md) | **한국어**

![image](https://i.imgur.com/qrYfsnh.png)


## 1. Project Overview
> **"What I cannot create, I do not understand."** - Richard Feynman

이 프로젝트는 **PyTorch의 `autograd` 엔진 없이**, 오직 `NumPy`만을 사용하여 딥러닝의 학습 과정(Forward, Backward, Optimizer)을 밑바닥부터 구현한 라이브러리입니다.
블랙박스로 여겨지던 딥러닝 프레임워크의 내부 동작 원리를 역공학(Reverse Engineering)하여, 계산 그래프(Computational Graph)와 역전파(Backpropagation)의 수학적 본질을 이해하는 것을 목표로 합니다.

## 2. Key Features
* **Pure NumPy Implementation:** `torch.autograd` 의존성 0%.
* **Modular Design:** `Layer` 기반의 객체 지향 설계.
* **Mathematical Rigor:** 연쇄 법칙(Chain Rule)에 기반한 정확한 기울기 계산.

## 3. Mathematical Foundations
이 라이브러리는 연쇄 법칙(Chain Rule)을 통해 국소적 미분(Local Gradient)을 상류(Upstream)로 전달합니다.

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}
$$

특히, `Softmax-with-Loss` 계층의 역전파는 다음과 같이 우아하게 유도됨을 코드 레벨에서 증명합니다.

$$
\frac{\partial L}{\partial z_k} = y_k - t_k
$$

(여기서 $y_k$는 소프트맥스 출력, $t_k$는 정답 레이블입니다.)

## 4. Verification
구현의 정확성은 다음 두 가지 방법으로 엄격하게 검증됩니다.
1. **Gradient Checking:** 수치 미분(Numerical Differentiation)과의 비교.
2. **Cross-Validation with PyTorch:** PyTorch의 자동 미분 결과와 $10^{-5}$ 이하의 오차 범위 내 일치 확인.
