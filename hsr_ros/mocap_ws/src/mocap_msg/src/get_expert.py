import rospy
from geometry_msgs.msg import PoseStamped
import pickle, datetime


class GetExpert():
    def __init__(self):
        self.food_list = (
            "tempura_satsumaimo",
            "tempura_pumpkin",
            "tempura_shishitou",
            "tempura_eggplant",
            "tempura_shrimp",
            "gripper",
        )

        self.exp_pose_dict = {key:PoseStamped() for key in self.food_list}
        self.current_pose_list = [PoseStamped() for i in range(len(self.food_list))]

        rospy.init_node('listener_node', anonymous=True)

        sub_list = []
        for i, food_name in enumerate(self.food_list):
            sub = rospy.Subscriber(
                "/mocap_pose_topic/{}_pose".format(food_name), PoseStamped, self.callback, callback_args=i)
            sub_list.append(sub)
        
        self.r = rospy.Rate(10)
        self.r.sleep()

    def callback(self, msg, id):
        self.current_pose_list[id] = msg.pose

    def main(self):
        dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        print(dt_now)

        for i in range(len(self.food_list)):
            food_name = self.food_list[i]
            try: 
                next_food_name = self.food_list[i+1]
            except Exception as e: 
                next_food_name = e

            self.r.sleep()

            pose = self.current_pose_list[i]
            self.exp_pose_dict[food_name] = pose
            print(self.exp_pose_dict)

            input(
                "Done: {} \n"
                "Next: {} \n"
                "(press Enter to next food)".format(food_name, next_food_name)
            )
        
        print('*'*100)
        print(self.exp_pose_dict)

        with open('exp{}.pkl'.format(dt_now), "wb") as f:
            pickle.dump(self.exp_pose_dict, f)


if __name__=="__main__":
    ge = GetExpert()
    ge.main()

