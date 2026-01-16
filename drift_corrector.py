import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
#import tf_transformations
import math

class HeadingLockCorrector(Node):
    def __init__(self):
        super().__init__('heading_lock_corrector')

        # --- TUNING PARAMETERS ---
        self.max_speed = 0.4    
        self.static_bias = -0.75  # Tuned for your high-speed drift
        self.ramp_duration = 1.0  # INCREASED: Matches hardware acceleration
        self.kp_yaw = 4.0         # LOWERED: Prevents initial engagement snap
        self.ki_yaw = 6.0         
        self.kd_yaw = 1.2         # INCREASED: Acts as a high-speed shock absorber
        self.i_clamp = 0.6     
        
        # --- STATE ---
        self.actual_vx = 0.0      
        self.target_yaw = None
        self.current_yaw = 0.0
        self.integral_error = 0.0
        self.last_yaw_error = 0.0
        self.is_driving_straight = False
        self.start_time = None
        self.last_time = self.get_clock().now()
        self.last_sign_vx = 0.0 
        

        self.sub_odom = self.create_subscription(Odometry, '/merged_odom', self.odom_callback, 10)
        self.sub_teleop = self.create_subscription(Twist, '/cmd_vel_teleop', self.teleop_callback, 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info("Heading Lock Corrector Active.")
    '''
    def get_yaw_from_quat(self, quat):
        q = [quat.x, quat.y, quat.z, quat.w]
        _, _, yaw = tf_transformations.euler_from_quaternion(q)
        return yaw
    '''
    def get_yaw_from_msg(self, q_msg):
        # Formula to extract Yaw from Quaternion (x, y, z, w)
        # q_msg is the message.pose.pose.orientation object
        x, y, z, w = q_msg.x, q_msg.y, q_msg.z, q_msg.w
        
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        return math.atan2(t3, t4)

    def odom_callback(self, msg):
        #self.current_yaw = self.get_yaw_from_quat(msg.pose.pose.orientation)
        self.current_yaw = self.get_yaw_from_msg(msg.pose.pose.orientation)
        self.actual_vx = msg.twist.twist.linear.x # Capture physical speed

    def teleop_callback(self, msg):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        out_msg = Twist()
        out_msg.linear.x = msg.linear.x

        user_wants_straight = abs(msg.angular.z) < 0.0001 and abs(msg.linear.x) > 0.01
        current_sign_vx = 1.0 if msg.linear.x > 0 else -1.0 if msg.linear.x < 0 else 0.0
        direction_changed = (current_sign_vx != self.last_sign_vx) and (current_sign_vx != 0)
        min_treshold = abs(self.actual_vx) > 0.05
        if user_wants_straight and min_treshold:
            if not self.is_driving_straight or direction_changed:
                self.target_yaw = self.current_yaw
                self.integral_error = 0.0
                self.is_driving_straight = True
                self.start_time = now
                self.last_sign_vx = current_sign_vx
                self.last_yaw_error = 0.0 
                return

            elapsed = (now - self.start_time).nanoseconds / 1e9
            speed_ratio = abs(msg.linear.x) / self.max_speed

            # 1. FF uses ACTUAL velocity to prevent pre-steering
            ramp_factor = min(elapsed / self.ramp_duration, 1.0)
            ff_term = (self.static_bias * self.actual_vx) * ramp_factor

            # 2. ERROR Calculation
            error = self.target_yaw - self.current_yaw
            error = math.atan2(math.sin(error), math.cos(error))

            if dt > 0 and dt < 0.5:
                self.integral_error += error * dt
            self.integral_error = max(min(self.integral_error, self.i_clamp), -self.i_clamp)
            
            # 3. SOFTENED D-TERM (Wait 0.2s to start D-term to prevent kick)
            if elapsed > 0.2 and dt > 0:
                derivative = (error - self.last_yaw_error) / dt
            else:
                derivative = 0.0
            
            # 4. FINAL AUTHORITY
            fb_term = (self.kp_yaw * error + self.ki_yaw * self.integral_error + self.kd_yaw * derivative) * (1.0 + speed_ratio * 1.5)
            
            out_msg.angular.z = ff_term + fb_term
            self.last_yaw_error = error
        
        else:
            if self.is_driving_straight:
                self.is_driving_straight = False
                self.target_yaw = None
                self.integral_error = 0.0
            out_msg.angular.z = msg.angular.z

        self.pub_cmd.publish(out_msg)


        
def main(args=None):
    rclpy.init(args=args)
    node = HeadingLockCorrector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()