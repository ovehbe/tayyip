import cv2
import numpy as np
import csv

# 1. Initialize webcam
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Could not open webcam."

# 2. ORB detector & BF matcher
orb = cv2.ORB_create(nfeatures=2000)
bf  = cv2.BFMatcher(cv2.NORM_HAMMING)

# 3. Camera intrinsics (dummy values)
K = np.array([[700,   0, 320],
              [  0, 700, 240],
              [  0,   0,   1]])

# 4. Pose state
cur_R = np.eye(3)
cur_t = np.zeros((3, 1))

# 5. Trajectory display canvas
traj_size = 600
trajectory = np.zeros((traj_size, traj_size, 3), dtype=np.uint8)

# 6. Previous-frame storage
prev_kp = None
prev_des = None

# 7. Prepare CSV recording
positions = []  # list of (frame_idx, x, y, z)
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)

    # Run VO if descriptors valid
    if prev_des is not None and des is not None \
       and prev_des.shape[1] == des.shape[1] \
       and len(prev_kp) > 10 and len(kp) > 10:

        matches = bf.match(prev_des, des)
        matches = sorted(matches, key=lambda m: m.distance)[:100]

        if len(matches) >= 8:
            pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
            pts2 = np.float32([   kp[m.trainIdx].pt for m in matches])

            E, _ = cv2.findEssentialMat(
                pts2, pts1, K,
                method=cv2.RANSAC, prob=0.999, threshold=1.0
            )
            if E is not None and E.shape == (3,3):
                _, R, t, _ = cv2.recoverPose(E, pts2, pts1, K)

                # accumulate pose
                cur_t += cur_R @ t
                cur_R = R @ cur_R

                # record and draw
                x, y, z = cur_t.flatten()
                positions.append((frame_idx, x, y, z))

                dx = int(x) + traj_size//2
                dy = int(z) + traj_size//3
                dx = np.clip(dx, 0, traj_size-1)
                dy = np.clip(dy, 0, traj_size-1)
                cv2.circle(trajectory, (dx, dy), 2, (0,255,0), 2)

                # overlay XYZ
                label = f"X:{x:.2f}  Y:{y:.2f}  Z:{z:.2f}"
                lp_x, lp_y = 10, 40
                cv2.rectangle(trajectory, (lp_x-5, lp_y-25), (lp_x+300, lp_y+5), (0,0,0), -1)
                cv2.putText(trajectory, label, (lp_x, lp_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    # Update for next frame
    prev_kp, prev_des = kp, des
    frame_idx += 1

    # Display windows
    cv2.imshow("Live Camera", frame)
    cv2.imshow("Trajectory", trajectory)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

# 8. Write CSV file
csv_path = "trajectory.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "x", "y", "z"]);
    writer.writerows(positions)

print(f"Saved {len(positions)} poses to '{csv_path}'")