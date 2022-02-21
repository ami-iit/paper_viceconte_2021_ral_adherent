# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import math
import numpy as np
from scenario import core
from typing import List, Dict
from dataclasses import dataclass
from gym_ignition.utils import misc
from scenario import gazebo as scenario
from adherent.MANN.utils import read_from_file
from adherent.data_processing.utils import iCub

import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt

# =====================
# MODEL INSERTION UTILS
# =====================

@dataclass
class SphereURDF:
    """Class for defining a sphere urdf with parametric radius and color."""

    radius: float = 0.5
    color: tuple = (1, 1, 1, 1)

    def urdf(self) -> str:
        i = 2.0 / 5 * 1.0 * self.radius * self.radius
        urdf = f"""
            <robot name="sphere_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

                <!-- ====== -->
                <!-- COLORS -->
                <!-- ====== -->
                <material name="custom">
                    <color rgba="{self.color[0]} {self.color[1]} {self.color[2]} {self.color[3]}"/>
                </material>
                <gazebo reference="sphere">
                    <visual>
                      <material>
                        <diffuse>{self.color[0]} {self.color[1]} {self.color[2]} {self.color[3]}</diffuse>
                      </material>
                    </visual>
                    <collision>
                        <surface>
                          <friction>
                            <ode>
                              <mu>0.0</mu>
                            </ode>
                          </friction>
                        </surface>
                    </collision>
                </gazebo>

                <!-- ===== -->
                <!-- LINKS -->
                <!-- ===== -->

                <link name="sphere">
                    <inertial>
                      <origin rpy="0 0 0" xyz="0 0 0"/>
                      <mass value="1.0"/>
                      <inertia ixx="{i}" ixy="0" ixz="0" iyy="{i}" iyz="0" izz="{i}"/>
                    </inertial>
                    <visual>
                      <geometry>
                        <sphere radius="{self.radius}"/>
                      </geometry>
                      <origin rpy="0 0 0" xyz="0 0 0"/>
                      <material name="custom">
                        <color rgba="{self.color[0]} {self.color[1]} {self.color[2]} {self.color[3]}"/>
                      </material>
                    </visual>
                    <collision>
                      <geometry>
                        <sphere radius="{self.radius}"/>
                      </geometry>
                      <origin rpy="0 0 0" xyz="0 0 0"/>
                    </collision>
                </link>
                <gazebo reference="sphere">
                  <collision>
                    <surface>
                      <friction>
                        <ode>
                          <mu>0.0</mu>
                          <mu2>0.0</mu2>
                        </ode>
                      </friction>
                    </surface>
                  </collision>
                </gazebo>
            </robot>"""

        return urdf

class Shape:
    """Helper class to simplify shape insertion."""

    def __init__(self,
                 world: scenario.World,
                 position: List[float] = (0, 0, 0),
                 orientation: List[float] = (1, 0, 0, 0),
                 model_string: str = SphereURDF(radius=0.02).urdf()):
        self.sdf = misc.string_to_file(model_string)

        # Assing incremental default name when multiple shapes are inserted
        name = scenario.get_model_name_from_sdf(self.sdf)
        index = 0
        while name in world.model_names():
            name = f"{name}{index}"

        # Insert the shape in the world
        assert world.insert_model(self.sdf, core.Pose(position, orientation), name)

        # Get and store the model and the world
        self.model = world.get_model(model_name=name)
        self.world = world

# =====================
# JOYSTICK DEVICE UTILS
# =====================

def quadratic_bezier(p0: np.array, p1: np.array, p2: np.array, t: np.array) -> List:
    """Define a discrete quadratic Bezier curve. Given the initial point p0, the control point p1 and
       the final point p2, the quadratic Bezier consists of t points and is defined by:
               Bezier(p0, p1, p2, t) = (1 - t)^2 p0 + 2t (1 - t) p1 + t^2 p2
    """

    quadratic_bezier = []

    for t_i in t:
        p_i = (1 - t_i) * (1 - t_i) * p0 + 2 * t_i * (1 - t_i) * p1 + t_i * t_i * p2
        quadratic_bezier.append(p_i)

    return quadratic_bezier

def compute_angle_wrt_x_positive_semiaxis(current_facing_direction: List) -> float:
    """Compute the angle between the current facing direction and the x positive semiaxis."""

    # Define the x positive semiaxis
    x_positive_semiaxis = np.asarray([1, 0])

    # Compute the yaw between the current facing direction and the world x axis
    cos_theta = np.dot(x_positive_semiaxis, current_facing_direction) # unitary norm vectors
    sin_theta = np.cross(x_positive_semiaxis, current_facing_direction) # unitary norm vectors
    angle = math.atan2(sin_theta, cos_theta)

    return angle

# ===========================
# TRAJECTORY GENERATION UTILS
# ===========================

def define_initial_nn_X(robot: str) -> List:
    """Define the robot-specific initial input X for the network used for trajectory generation."""

    if robot != "iCubV2_5":
        raise Exception("Initial network input X only defined for iCubV2_5.")

    # Initial input manually retrieved from a standing pose
    initial_nn_X = [[0.2119528955450237, -0.0030393414305661757, 0.2088665596830631, -0.0021020581878808007,
                     0.20531231168918174, -0.0008621646186016302, 0.2038031784172399, 0.0006000018873639539,
                     0.20492310200003655, 0.00219387868363895, 0.0, 0.0, -0.2861243642562259, 0.009288430936687906,
                     -0.31823372134674327, 0.0041163061304422335, -0.3351330296979504, 0.00013665165796125482,
                     -0.32760743228415534, 0.0013394518793597135, -0.28954434529500506, -0.002034172225900831,
                     -0.22271595678165187, -0.007108432655308732, 0.4874557110316117, -0.060262978658173996,
                     0.483219430286254, -0.05549580293883961, 0.48355811354675093, -0.05073190069674229,
                     0.48430037825453315, -0.0463878541965186, 0.47622379318033997, -0.042673799121161156, 0.0,
                     -0.001670958751021921, 0.48823702730673707, -0.012628077673047147, 0.4985278001448309,
                     -0.017930827973037217, 0.4894521106023875, -0.013467714394521855, 0.48597445992611943,
                     -0.004418383745209343, 0.4891737721962037, -0.0006736504004182218, 0.4998745767668644,
                     -6.208532325985457e-05, -0.14660114753459738, -0.013635038569482734, -0.14972572660656353,
                     -0.013889675494643422, -0.14485898865197464, -0.014291839679283299, -0.13868430844742163,
                     -0.015641834593965512, -0.14136717424413048, -0.017874811642905507, -0.15142562459589684,
                     -0.020944871631760793, -0.07793121543356057, -0.002198970320957779, 0.12319525161531483,
                     0.002993875070530926, -1.1538873156781917, 0.01723559513698802, -1.495295660555109,
                     0.0010038652592341854, -1.8281665557305795, 0.0006766967160280221, -2.159682389892566,
                     -0.006655788893852345, -2.4211997845061326, -0.9981968, -0.3760584, 0.37910417, 0.65148425,
                     0.3296967, 0.40070292, -1.0328596, -0.36308038, 0.41093844, 0.64171183, 0.28167534, 0.3818131,
                     -1.9024365, 0.00088662654, 0.0018205047, 0.35245338, -0.33301216, -0.21851036, -0.73831433,
                     0.61941016, 0.8847597, -0.98411304, -0.5619794, 0.03115137, -0.091870524, 0.35733885,
                     -0.366953, -0.19864298, -0.71252215, 0.70668554, 0.8847598, -0.98411304, 0.0064472854,
                     -0.003662251, -0.027689233, -0.019645333, -0.010648936, -0.00738357, 0.015316837, 0.04705859,
                     0.011168551, -0.013807453, -0.0048871785, -0.027151093, -0.12258834, -0.028669998, 0.04093896,
                     -0.0797086, -0.010067629, 0.10308291, 0.07587971, -0.0666933, -0.018514698, 0.019428264,
                     0.0027377363, 0.042420693, -0.009432776, 0.048711963, -0.0032096095, -0.0069472883,
                     0.016040523, 0.0183498, -0.01851473, 0.019428236]]

    return initial_nn_X

def define_initial_past_trajectory(robot: str) -> (List, List, List):
    """Define the robot-specific initialization of the past trajectory data used for trajectory generation."""

    if robot != "iCubV2_5":
        raise Exception("Initial past trajectory data only defined for iCubV2_5.")

    # The above quantities are expressed in the frame specified by the initial base position and the facing direction

    # Initial past base positions manually retrieved from a standing pose
    initial_past_trajectory_base_pos = [[-0.0006781887991487454, 0.0005382023957937025],
                                [-0.0006647239425803457, 0.0005273088163705415],
                                [-0.0006512643546792456, 0.0005164206039375526],
                                [-0.0006377953293649197, 0.0005055375639224852],
                                [-0.0006243223759395536, 0.0004946598636862636],
                                [-0.0006108399997998801, 0.0004837873556333961],
                                [-0.0005973672439964133, 0.0004729208481432019],
                                [-0.0005838774043584994, 0.000462058202637307],
                                [-0.0005703864089978013, 0.00045120053923471386],
                                [-0.0005568976029413277, 0.00044034668352684476],
                                [-0.0005433944044652785, 0.00042950336076639064],
                                [-0.0005298872456197738, 0.00041866148475417985],
                                [-0.0005163891602592688, 0.0004078273201884596],
                                [-0.0005028772288014707, 0.0003969962242274589],
                                [-0.0004893568206573056, 0.0003861719082838056],
                                [-0.0004758425651066602, 0.0003753518627324191],
                                [-0.00046231312340075495, 0.00036453423210500597],
                                [-0.00044878604841946013, 0.00035372630074053177],
                                [-0.0004352530747953605, 0.00034292112934518164],
                                [-0.00042172149782268163, 0.00033212353566379113],
                                [-0.0004081797610404143, 0.0003213324600119949],
                                [-0.0003946302109381543, 0.0003105427764365998],
                                [-0.00038108095663678086, 0.000299761652046352],
                                [-0.0003675236365612126, 0.0002889855808213132],
                                [-0.0003539697410959713, 0.0002782137851076101],
                                [-0.00034041768049816534, 0.0002674457780258597],
                                [-0.00032684196379834704, 0.000256685071996519],
                                [-0.00031326716933113526, 0.0002459287909804182],
                                [-0.00029969788590518485, 0.00023517725075197912],
                                [-0.00028611917067016793, 0.0002244313688890221],
                                [-0.0002725307912706841, 0.00021369435529300127],
                                [-0.00025894933834369005, 0.00020295593849716727],
                                [-0.00024536710778585904, 0.00019222322705029503],
                                [-0.000231767236595024, 0.00018150092194441208],
                                [-0.00021815871347549421, 0.00017077920312574124],
                                [-0.00020455895870665294, 0.00016006834116839483],
                                [-0.00019095141496312167, 0.00014936037600883187],
                                [-0.0001773345902858732, 0.00013865422551428646],
                                [-0.00016373661066682184, 0.0001279587235979194],
                                [-0.00015009695166202328, 0.00011726740411708459],
                                [-0.00013648127926164732, 0.0001065802552652773],
                                [-0.00012284291258793977, 9.589807810900453e-05],
                                [-0.00010921928546949104, 8.522111760777935e-05],
                                [-9.558259372471922e-05, 7.454690983824729e-05],
                                [-8.193676260565677e-05, 6.388213666527646e-05],
                                [-6.829031781475642e-05, 5.3223206770590906e-05],
                                [-5.4640516832067336e-05, 4.256765707546707e-05],
                                [-4.099221572739473e-05, 3.191479359366577e-05],
                                [-2.733307550666302e-05, 2.1273139149335884e-05],
                                [-1.3662890395487173e-05, 1.063469931815647e-05],
                                [0.0, 0.0]]

    # Initial past facing directions manually retrieved from a standing pose
    initial_past_trajectory_facing_dirs = [[0.9998092990492257, -0.019528582506055422],
                                    [0.9998168506034699, -0.01913805761718488],
                                    [0.9998242495649122, -0.018747532689931887],
                                    [0.9998314960180585, -0.018357003303677545],
                                    [0.9998385899327197, -0.01796647103220058],
                                    [0.9998455313288402, -0.0175759347333077],
                                    [0.9998523201559695, -0.017185397252453403],
                                    [0.9998589564711968, -0.016794855293497706],
                                    [0.9998654401899228, -0.01640431387812107],
                                    [0.9998717713630115, -0.01601377005560508],
                                    [0.9998779500423899, -0.015623220507573019],
                                    [0.9998839761713595, -0.015232668700922845],
                                    [0.9998898497279782, -0.014842116121402793],
                                    [0.9998955707342037, -0.014451561373105574],
                                    [0.99990113916948, -0.014061005923328785],
                                    [0.9999065550758257, -0.013670446824995854],
                                    [0.9999118184095479, -0.013279887232614156],
                                    [0.9999169292288519, -0.012889322772870165],
                                    [0.9999218874586415, -0.01249875918429945],
                                    [0.9999266931712342, -0.012108190766611501],
                                    [0.9999313462939388, -0.011717623427582714],
                                    [0.9999358468605792, -0.011327054481003577],
                                    [0.9999401948901391, -0.010936482207298913],
                                    [0.9999443903613943, -0.010545908437858902],
                                    [0.9999484332728986, -0.010155333331557733],
                                    [0.99995232364119, -0.009764755223974861],
                                    [0.999956061452534, -0.009374175394966032],
                                    [0.9999596466925486, -0.008983595411236894],
                                    [0.99996307938682, -0.008593012465219062],
                                    [0.9999663595078725, -0.008202429674901989],
                                    [0.9999694870684395, -0.007811845625800506],
                                    [0.999972462068608, -0.007421260300351187],
                                    [0.9999752844950315, -0.007030675577780891],
                                    [0.9999779543599308, -0.006640089918632473],
                                    [0.9999804716745225, -0.00624950154805433],
                                    [0.9999828364186417, -0.005858913562069191],
                                    [0.9999850486165925, -0.005468321796556921],
                                    [0.9999871082497419, -0.005077729248250766],
                                    [0.9999890153097547, -0.004687137700856822],
                                    [0.999990769812952, -0.004296543831877126],
                                    [0.9999923717399964, -0.0039059521011293643],
                                    [0.9999938211140119, -0.003515356852062973],
                                    [0.9999951179123994, -0.0031247642097102634],
                                    [0.999996262155962, -0.0027341678997491606],
                                    [0.9999972538295608, -0.002343572771777715],
                                    [0.9999980929345732, -0.0019529790618736484],
                                    [0.9999987794762837, -0.0015623846973456307],
                                    [0.999999313457387, -0.0011717869918159319],
                                    [0.999999694869178, -0.000781192390328727],
                                    [0.9999999237179578, -0.0003905945192233482],
                                   [1.0, 0.0]]

    # Initial past base velocities manually retrieved from a standing pose
    initial_past_trajectory_base_vel = [[0.0026596026673994097, -0.0036648600695725142],
                                [0.0026610339452683906, -0.003663820959069162],
                                [0.0026624618469103353, -0.0036627708065018845],
                                [0.0026638952851042526, -0.0036617410596578807],
                                [0.0026653254955963652, -0.003660689836623215],
                                [0.0026667550716268367, -0.003659653708431181],
                                [0.0026681811158322005, -0.003658616979253869],
                                [0.002669616214903244, -0.0036575745998335073],
                                [0.0026710509070692248, -0.0036565316572672855],
                                [0.00267246954438113, -0.0036554879581719886],
                                [0.0026738972878064464, -0.0036544333858607716],
                                [0.0026753244934117526, -0.0036533886916647177],
                                [0.0026767511738523863, -0.003652353875415824],
                                [0.0026781776799157907, -0.0036512976312858153],
                                [0.002679616178485772, -0.003650251396067544],
                                [0.0026810261025279327, -0.0036492043111861438],
                                [0.0026824668620947442, -0.0036481622020557683],
                                [0.002683879098004333, -0.0036471140460612194],
                                [0.002685312832378327, -0.0036460655370843115],
                                [0.0026867273789002856, -0.00364501630574531],
                                [0.0026881603360259347, -0.0036439614591838782],
                                [0.002689586538764857, -0.00364291643731224],
                                [0.0026910092820887346, -0.003641860398506238],
                                [0.0026924221129044816, -0.003640819393236157],
                                [0.0026938439971073334, -0.003639767466126147],
                                [0.002695268663612408, -0.003638704567407736],
                                [0.002696689721876551, -0.0036376515283696865],
                                [0.0026981009249859673, -0.0036366083190739903],
                                [0.002699521191342563, -0.0036355489551039145],
                                [0.0027009441479402046, -0.0036344942684490705],
                                [0.002702366670581848, -0.0036334442428568987],
                                [0.002703779416296505, -0.0036323884065143142],
                                [0.0027051980439246415, -0.003631321606572732],
                                [0.0027066193482793745, -0.003630264698469278],
                                [0.002708037110378657, -0.0036292072255858475],
                                [0.0027094638492328217, -0.0036281492193542698],
                                [0.0027108713957011837, -0.0036270906183057537],
                                [0.00271228165806409, -0.0036260314759179227],
                                [0.0027137040246006613, -0.003624971799103914],
                                [0.0027151228439162086, -0.0036239167802755035],
                                [0.002716531865543588, -0.0036228612023171043],
                                [0.002717952993942073, -0.003621799857935674],
                                [0.0027193737078339475, -0.0036207379559828126],
                                [0.002720775219705167, -0.003619665077193461],
                                [0.00272219510617805, -0.0036186072862108534],
                                [0.00272361144244023, -0.0036175437258978145],
                                [0.002725027353731481, -0.003616474392224524],
                                [0.0027264366238392225, -0.0036154201740137624],
                                [0.0027278548303872976, -0.0036143497261717085],
                                [0.0027292570019628926, -0.003613289208665431],
                                [0.0027306743709001454, -0.003612217655202784]]

    return initial_past_trajectory_base_pos, initial_past_trajectory_facing_dirs, initial_past_trajectory_base_vel

def define_initial_base_height(robot: str) -> List:
    """Define the robot-specific initial height of the base frame."""

    if robot != "iCubV2_5":
        raise Exception("Initial base height only defined for iCubV2_5.")

    initial_base_height = 0.6354

    return initial_base_height

def define_initial_base_yaw(robot: str) -> List:
    """Define the robot-specific initial base yaw expressed in the world frame."""

    if robot != "iCubV2_5":
        raise Exception("Initial base yaw only defined for iCubV2_5.")

    # For iCubV2_5, the initial base yaw is 180 degs since the x axis of the base frame points backward
    initial_base_yaw = math.pi

    return initial_base_yaw

def trajectory_blending(a0: List, a1: List, t: np.array, tau: float) -> List:
    """Blend the vectors a0 and a1 via:
           Blend(a0, a1, t, tau) = (1 - t^tau) a0 + t^tau a1
       Increasing tau means biasing more towards a1.
    """

    blended_trajectory = []

    for i in range(len(t)):
        p_i = (1 - math.pow(t[i], tau)) * np.array(a0[i]) + math.pow(t[i], tau) * np.array(a1[i])
        blended_trajectory.append(p_i.tolist())

    return blended_trajectory

def load_component_wise_input_mean_and_std(datapath: str) -> (Dict, Dict):
    """Compute component-wise input mean and standard deviation."""

    # Full-input mean and std
    Xmean = read_from_file(datapath + 'X_mean.txt')
    Xstd = read_from_file(datapath + 'X_std.txt')

    # Remove zeroes from Xstd
    for i in range(Xstd.size):
        if Xstd[i] == 0:
            Xstd[i] = 1

    # Retrieve component-wise input mean and std (used to normalize the next input for the network)
    Xmean_dict = {"past_base_positions": Xmean[0:12]}
    Xstd_dict = {"past_base_positions": Xstd[0:12]}
    Xmean_dict["future_base_positions"] = Xmean[12:24]
    Xstd_dict["future_base_positions"] = Xstd[12:24]
    Xmean_dict["past_facing_directions"] = Xmean[24:36]
    Xstd_dict["past_facing_directions"] = Xstd[24:36]
    Xmean_dict["future_facing_directions"] = Xmean[36:48]
    Xstd_dict["future_facing_directions"] = Xstd[36:48]
    Xmean_dict["past_base_velocities"] = Xmean[48:60]
    Xstd_dict["past_base_velocities"] = Xstd[48:60]
    Xmean_dict["future_base_velocities"] = Xmean[60:72]
    Xstd_dict["future_base_velocities"] = Xstd[60:72]
    Xmean_dict["future_traj_length"] = Xmean[72]
    Xstd_dict["future_traj_length"] = Xstd[72]
    Xmean_dict["s"] = Xmean[73:105]
    Xstd_dict["s"] = Xstd[73:105]
    Xmean_dict["s_dot"] = Xmean[105:]
    Xstd_dict["s_dot"] = Xstd[105:]

    return Xmean_dict, Xstd_dict

def load_output_mean_and_std(datapath: str) -> (List, List):
    """Compute output mean and standard deviation."""

    # Full-output mean and std
    Ymean = read_from_file(datapath + 'Y_mean.txt')
    Ystd = read_from_file(datapath + 'Y_std.txt')

    # Remove zeroes from Ystd
    for i in range(Ystd.size):
        if Ystd[i] == 0:
            Ystd[i] = 1

    return Ymean, Ystd

# ===================
# VISUALIZATION UTILS
# ===================

def visualize_generated_motion(icub: iCub,
                               gazebo: scenario.GazeboSimulator,
                               posturals: Dict,
                               raw_data: List,
                               blending_coeffs: Dict,
                               plot_blending_coeffs: bool) -> None:
    """Visualize the generated motion along with the joystick inputs used to generate it and,
    optionally, the activations of the blending coefficients during the trajectory generation."""

    # Retrieve joint and base posturals
    joint_posturals = posturals["joints"]
    base_posturals = posturals["base"]

    # Retrieve blending coefficients
    if plot_blending_coeffs:
        w_1 = blending_coeffs["w_1"]
        w_2 = blending_coeffs["w_2"]
        w_3 = blending_coeffs["w_3"]
        w_4 = blending_coeffs["w_4"]

    # Define controlled joints
    controlled_joints = icub.joint_names()

    # Plot configuration
    plt.ion()

    for frame_idx in range(len(joint_posturals)):

        # Debug
        print(frame_idx, "/", len(joint_posturals))

        # ======================
        # VISUALIZE ROBOT MOTION
        # ======================

        # Retrieve the current joint positions
        joint_postural = joint_posturals[frame_idx]
        joint_positions = [joint_postural[joint] for joint in controlled_joints]

        # Retrieve the current base position and orientation
        base_postural = base_posturals[frame_idx]
        base_position = base_postural['postion']
        base_quaternion = base_postural['wxyz_quaternions']

        # Reset the robot configuration in the simulator
        icub.to_gazebo().reset_base_pose(base_position, base_quaternion)
        icub.to_gazebo().reset_joint_positions(joint_positions, controlled_joints)
        gazebo.run(paused=True)

        # =====================================
        # PLOT THE MOTION DIRECTION ON FIGURE 1
        # =====================================

        # Retrieve the current motion direction
        curr_raw_data = raw_data[frame_idx]
        curr_x = curr_raw_data[0]
        curr_y = curr_raw_data[1]

        plt.figure(1)
        plt.clf()

        # Circumference of unitary radius
        r = 1
        x = np.linspace(-r, r, 1000)
        y = np.sqrt(-x ** 2 + r ** 2)
        plt.plot(x, y, 'r')
        plt.plot(x, -y, 'r')

        # Motion direction
        plt.scatter(0, 0, c='r')
        desired_motion_direction = plt.arrow(0, 0, curr_x, -curr_y, length_includes_head=True, width=0.01,
                                             head_width=8 * 0.01, head_length=1.8 * 8 * 0.01, color='r')

        # Plot configuration
        plt.axis('scaled')
        plt.xlim([-1.2, 1.2])
        plt.ylim([-1.4, 1.2])
        plt.axis('off')
        plt.legend([desired_motion_direction], ['DESIRED MOTION DIRECTION'], loc="lower center")

        # =====================================
        # PLOT THE FACING DIRECTION ON FIGURE 2
        # =====================================

        # Retrieve the current facing direction
        curr_z = curr_raw_data[2]
        curr_rz = curr_raw_data[3]

        plt.figure(2)
        plt.clf()

        # Circumference of unitary norm
        r = 1
        x = np.linspace(-r, r, 1000)
        y = np.sqrt(-x ** 2 + r ** 2)
        plt.plot(x, y, 'b')
        plt.plot(x, -y, 'b')

        # Facing direction
        plt.scatter(0, 0, c='b')
        desired_facing_direction = plt.arrow(0, 0, curr_z, -curr_rz, length_includes_head=True, width=0.01,
                                             head_width=8 * 0.01, head_length=1.8 * 8 * 0.01, color='b')

        # Plot configuration
        plt.axis('scaled')
        plt.xlim([-1.2, 1.2])
        plt.ylim([-1.4, 1.2])
        plt.axis('off')
        plt.legend([desired_facing_direction], ['DESIRED FACING DIRECTION'], loc="lower center")

        # ==========================================
        # PLOT THE BLENDING COEFFICIENTS ON FIGURE 3
        # ==========================================

        if plot_blending_coeffs:
            # Retrieve the blending coefficients up to the current time
            curr_w_1 = w_1[:frame_idx]
            curr_w_2 = w_2[:frame_idx]
            curr_w_3 = w_3[:frame_idx]
            curr_w_4 = w_4[:frame_idx]

            plt.figure(3)
            plt.clf()

            plt.plot(range(len(curr_w_1)), curr_w_1, 'r')
            plt.plot(range(len(curr_w_2)), curr_w_2, 'b')
            plt.plot(range(len(curr_w_3)), curr_w_3, 'g')
            plt.plot(range(len(curr_w_4)), curr_w_4, 'y')

            # Plot configuration
            plt.title("Blending coefficients profiles")
            plt.xlim([0, len(w_1)])
            plt.ylabel("Blending coefficients")
            plt.xlabel("Time [s]")

        # Plot
        plt.show()
        plt.pause(0.0001)

    input("Press Enter to end the visualization of the generated trajectory.")
