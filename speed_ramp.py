import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class SpeedRamp(Node):
    def __init__(self):
        super().__init__('speed_ramp_node')
        
        self.pub = self.create_publisher(Twist, '/cmd_vel_teleop', 10)
        
        self.max_speed = 0.4
        self.step_size = 0.01
        self.step_duration = 2.0  # Seconds per step
        
        self.current_speed = 0.0
        self.last_step_time = time.time()
        self.start_time = time.time()
        
        # We must publish at 20Hz to keep the robot's watchdog happy
        self.create_timer(0.05, self.control_loop)
        
        print(f"RAMP STARTED: 0.0 -> {self.max_speed} m/s over ~80 seconds.")

    def control_loop(self):
        now = time.time()
        
        # 1. Check if it is time to increase speed
        if now - self.last_step_time >= self.step_duration:
            if self.current_speed < self.max_speed:
                self.current_speed += self.step_size
                # Round to 2 decimals to avoid floating point ugliness (0.30000004)
                self.current_speed = round(self.current_speed, 2)
                self.last_step_time = now
                print(f"Speed UP: {self.current_speed} m/s")
            else:
                print("Max Speed Reached. Holding...")
                # Optional: Uncomment to stop automatically after reaching max
                self.current_speed = 0.0
                raise SystemExit
        
        # 2. Publish the continuous command
        msg = Twist()
        msg.linear.x = self.current_speed
        msg.angular.z = 0.0  # Drift Corrector will lock heading here
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = SpeedRamp()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Safety Stop
        stop_msg = Twist()
        node.pub.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()