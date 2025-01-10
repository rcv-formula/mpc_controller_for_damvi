안녕하세요 MPC 부분을 맡은 임준형입니다

<01.08.2025 기준>
MPC Controller - Not Tested
ptcld2scan - Work on progress
scan2cost - Finished

scan2cost Package
말 그대로, 2d lidar 의 /scan 토픽을 받아서 local_costmap을 출력해줍니다.
local_costmap은 base_link frame 기준으로 생성됩니다. 따라서 laser frame의 /scan 과 겹쳐서 rviz2에 시각화하면, x축으로 -0.2m 만큼 /scan 데이터와 차이가 납니다.
static_transform 값을 수정하여 이 값을 조절할 수 있습니다. 나중에 tf calibration을 진행한 후, 해당 값을 입력해주면 될듯 합니다. 

config.yaml 파일에서 설정 가능한 것 :
input topic, output topic 이름 변경 가능
base frame 변경 가능
map resolution, origin_x, origin_y 단위 [m]
width, heigt 단위 [cell]

처음 실행 시:
scan2cost 패키지를 복사를 하던, git clone을 하던 알아서 workspace에 넣습니다.
ros2_ws/src/scan2cost
ros2_ws/src 에서 rosdep install --from-paths . --ignore-src -r -y 실행 해줍니다.
config 수정을 했다면, colcon build --packages-up-to scan2cost 를 실행합니다.
source install/setup.bash 소싱해주고
ros2 launch scan2cost scan2cost.launch.py 를 실행합니다.

static_transform 은 base_link 와 laser 사이의 tf 관계입니다.

MPC Controller
Ackermann Steering - Bicycle Model
Constraints : 후진 불가, Config에 입력된 최대 속도 및 각도, Model Constraints

ptcld2scan
input frame 도 반영할 수 있게 변경 필요
tf time sync 센서간 time 동기화 시켜보기

[To-Do]
{scan2cost} tf 토픽을 받아와서 자동으로 base_link 와 laser 사이의 관계 파악 (나중에 통합할 때 하나의 config에 합칠때 그냥 불러오도록 해도 됨)
{ptcld2scan} tf time sync 문제 해결해야함.
{mpc} 흑담비에서, F1tenth Stack 기본 패키지 돌려서, /scan, /odom + simulated global path 로 MPC 테스트 해보기.
