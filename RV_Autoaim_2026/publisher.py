import rclpy
import threading
from rclpy.node import Node
from interface.msg import AutoaimData


class NodePublisher(Node):
    def __init__(self, autoaim_instance):
        super().__init__('autoaim_publisher')
        self.autoaim = autoaim_instance
        self.running = True
        self.last_output_seq = -1
        self.publisher_ = self.create_publisher(
             AutoaimData, 
             'autoaim_topic', 
             10
             )
        self.publish_thread = threading.Thread(target=self.publish_loop, daemon=True)
        self.publish_thread.start()
        self.get_logger().info('Autoaim Publisher Node started.')

    def publish_control(self, yaw, pitch, fire, source_timestamp):

        msg = AutoaimData()
        msg.yaw_angle_diff = float(yaw)
        msg.pitch_angle_diff = float(pitch)
        msg.fire = int(fire)
        msg.source_timestamp = float(source_timestamp)

        self.publisher_.publish(msg)

    def publish_loop(self):
        while self.running:
            yaw, pitch, fire, output_seq, _, source_timestamp = self.autoaim.wait_for_control_update(
                self.last_output_seq,
                timeout=0.1
            )

            if not self.running:
                break

            if output_seq == self.last_output_seq:
                continue

            self.last_output_seq = output_seq
            self.publish_control(yaw, pitch, fire, source_timestamp)
        # self.get_logger().info(
        #     f'Publishing: yaw_angle_diff={msg.yaw_angle_diff}, '
        #     f'pitch_angle_diff={msg.pitch_angle_diff}, fire={msg.fire}'
        # )

    def destroy_node(self):
        self.running = False
        try:
            with self.autoaim.output_condition:
                self.autoaim.output_condition.notify_all()
        except Exception:
            pass

        if hasattr(self, 'publish_thread') and self.publish_thread.is_alive():
            self.publish_thread.join(timeout=0.5)

        super().destroy_node()

def main(args=None):
        rclpy.init(args=args) 
        node = NodePublisher() 
        rclpy.spin(node)
        rclpy.shutdown() 

if __name__ == '__main__':
    main()
