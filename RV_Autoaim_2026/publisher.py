import rclpy
from rclpy.node import Node
from interface.msg import AutoaimData

class NodePublisher(Node):
    def __init__(self, autoaim_instance):
        super().__init__('autoaim_publisher')
        self.autoaim = autoaim_instance
        self.publisher_ = self.create_publisher(
             AutoaimData, 
             'autoaim_topic', 
             10
             )
        self.timer = self.create_timer(0.01, self.timer_callback)  # 每秒发布一次
        self.get_logger().info('Autoaim Publisher Node started.')

    def timer_callback(self):
        
        msg = AutoaimData()
        msg.yaw_angle_diff = float(self.autoaim.smooth_yaw)
        msg.pitch_angle_diff = float(self.autoaim.smooth_pitch)
        msg.fire = int(self.autoaim.fire_command)

        self.publisher_.publish(msg)
        # self.get_logger().info(f'Publishing: yaw_angle_diff={msg.yaw_angle_diff}, pitch_angle_diff={msg.pitch_angle_diff}, fire={msg.fire}')

def main(args=None):
        rclpy.init(args=args) 
        node = NodePublisher() 
        rclpy.spin(node)
        rclpy.shutdown() 

if __name__ == '__main__':
    main()