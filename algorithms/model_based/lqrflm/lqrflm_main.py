# coded by St.Watermelon (patched for Gymnasium)

import gymnasium as gym
from lqrflm_agent import LQRFLMagent
import math
import numpy as np
from config import configuration
import os
import time

def as_vec(obs):
    return np.asarray(obs, dtype=np.float64).squeeze()

def main():
    MAX_ITER = 60
    T = configuration["T"]

    # 저장 폴더 보장
    os.makedirs("./save_weights", exist_ok=True)

    # 1) 학습용 env (렌더링 없음)
    env_train = gym.make("Pendulum-v1")
    agent = LQRFLMagent(env_train)
    agent.update(MAX_ITER)
    env_train.close()

    # 학습된 게인
    Kt = agent.prev_control_data.Kt
    kt = agent.prev_control_data.kt

    print("\n\n Now play ................")

    # 2) 플레이용 env (렌더링 있음)
    env = gym.make("Pendulum-v1", render_mode="human")

    # 초기 상태(관측) 기준
    x0 = as_vec(agent.init_state)

    play_iter = 5
    save_gain = []

    for pn in range(play_iter):
        print("     play number :", pn + 1)

        # 초기 상태를 x0 근처로 맞추는 루프
        if pn < 2:
            bad_init = True
            while bad_init:
                obs, info = env.reset()
                state = as_vec(obs)

                x0err = state - x0
                if np.sqrt(x0err.T.dot(x0err)) < 0.1:
                    bad_init = False
        else:
            obs, info = env.reset()
            state = as_vec(obs)

        # NOTE: Kt/kt 길이가 T+1인지 보장 안 되면 range(T)로 바꿔야 함
        for t in range(T):
            # 행동 계산
            action = Kt[t, :, :].dot(state) + kt[t, :]
            action = np.clip(action, -agent.action_bound, agent.action_bound)

            # Pendulum action shape=(1,) 맞추기
            action = np.asarray(action, dtype=np.float32).reshape(1,)

            ang = math.atan2(state[1], state[0])
            print("Time:", t, ", angle:", ang * 180.0 / np.pi, "action:", action)

            save_gain.append([t, Kt[t, 0, 0], Kt[t, 0, 1], Kt[t, 0, 2], kt[t, 0]])

            obs, reward, terminated, truncated, info = env.step(action)
            state = as_vec(obs)

            if terminated or truncated:
                break

            # 속도 줄이고 싶으면 주석 해제
            # time.sleep(0.03)

    np.savetxt("./save_weights/kalman_gain.txt", save_gain)
    env.close()

if __name__ == "__main__":
    main()