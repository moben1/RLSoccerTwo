using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;

public enum Team
{
    Blue = 0,
    Purple = 1
}

public class AgentSoccer : Agent
{
    // Note that that the detectable tags are different for the blue and purple teams. The order is
    // * ball
    // * own goal
    // * opposing goal
    // * wall
    // * own teammate
    // * opposing player

    public enum Position
    {
        Striker,
        Goalie,
        Generic
    }

    [HideInInspector]
    public Team team;
    float m_KickPower;
    // The coefficient for the reward for colliding with a ball. Set using curriculum.
    float m_BallTouch;
    public Position position;

    const float k_Power = 2000f;
    float m_Existential;
    float m_LateralSpeed;
    float m_ForwardSpeed;


    [HideInInspector]
    public Rigidbody agentRb;
    SoccerSettings m_SoccerSettings;
    BehaviorParameters m_BehaviorParameters;
    public Vector3 initialPos;
    public float rotSign;

    public Vector2 ownGoalPos;
    public Vector2 opposingGoalPos;

    EnvironmentParameters m_ResetParams;

    public override void Initialize()
    {
        SoccerEnvController envController = GetComponentInParent<SoccerEnvController>();
        if (envController != null)
        {
            m_Existential = 1f / envController.MaxEnvironmentSteps;
        }
        else
        {
            m_Existential = 1f / MaxStep;
        }

        m_BehaviorParameters = gameObject.GetComponent<BehaviorParameters>();
        if (m_BehaviorParameters.TeamId == (int)Team.Blue)
        {
            team = Team.Blue;
            initialPos = new Vector3(transform.position.x - 5f, .5f, transform.position.z);
            rotSign = 1f;
        }
        else
        {
            team = Team.Purple;
            initialPos = new Vector3(transform.position.x + 5f, .5f, transform.position.z);
            rotSign = -1f;
        }
        if (position == Position.Goalie)
        {
            m_LateralSpeed = 1.0f;
            m_ForwardSpeed = 1.0f;
        }
        else if (position == Position.Striker)
        {
            m_LateralSpeed = 0.3f;
            m_ForwardSpeed = 1.3f;
        }
        else
        {
            m_LateralSpeed = 0.3f;
            m_ForwardSpeed = 1.0f;
        }
        m_SoccerSettings = FindObjectOfType<SoccerSettings>();
        agentRb = GetComponent<Rigidbody>();
        agentRb.maxAngularVelocity = 500;

        m_ResetParams = Academy.Instance.EnvironmentParameters;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Add observations
        SoccerEnvController envController = GetComponentInParent<SoccerEnvController>();
        // * Agent position
        sensor.AddObservation(this.transform.localPosition.x);
        sensor.AddObservation(this.transform.localPosition.z);
        // * Agent velocity
        sensor.AddObservation(this.agentRb.velocity.x);
        sensor.AddObservation(this.agentRb.velocity.z);
        // * Mate position
        Agent mate = envController.getMate(this);
        sensor.AddObservation(mate.transform.localPosition.x);
        sensor.AddObservation(mate.transform.localPosition.z);
        // * Opponents positions
        SimpleMultiAgentGroup opponents = envController.getOpponents(this.team);
        Debug.Assert(opponents.GetRegisteredAgents().Count == 2);
        foreach (Agent opponent in opponents.GetRegisteredAgents())
        {
            sensor.AddObservation(opponent.transform.localPosition.x);
            sensor.AddObservation(opponent.transform.localPosition.z);
        }
        // * Ball position
        sensor.AddObservation(envController.ball.transform.localPosition.x);
        sensor.AddObservation(envController.ball.transform.localPosition.z);
        // * Ball velocity
        sensor.AddObservation(envController.ballRb.velocity.x);
        sensor.AddObservation(envController.ballRb.velocity.z);
        // * Own goal position
        sensor.AddObservation(ownGoalPos.x);
        sensor.AddObservation(ownGoalPos.y);
        // * Opponent goal position
        sensor.AddObservation(opposingGoalPos.x);
        sensor.AddObservation(opposingGoalPos.y);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)

    {

        var actionZ = 2f * Mathf.Clamp(actionBuffers.ContinuousActions[0], -1f, 1f);
        var actionX = 2f * Mathf.Clamp(actionBuffers.ContinuousActions[1], -1f, 1f);
        var actionRotate = 2f * Mathf.Clamp(actionBuffers.ContinuousActions[2], -1f, 1f);

        if (position == Position.Goalie)
        {
            // Existential bonus for Goalies.
            AddReward(m_Existential);
        }
        else if (position == Position.Striker)
        {
            // Existential penalty for Strikers
            AddReward(-m_Existential);
        }

        gameObject.transform.Rotate(0f, actionRotate, 0f, Space.Self);
        agentRb.AddForce(transform.forward * m_ForwardSpeed * actionZ, ForceMode.VelocityChange);
        agentRb.AddForce(transform.right * m_LateralSpeed * actionX, ForceMode.VelocityChange);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var contActionsOut = actionsOut.ContinuousActions;
        contActionsOut[0] = Input.GetAxis("Vertical");
        contActionsOut[1] = Input.GetAxis("Horizontal");
        contActionsOut[2] = 0f;
        if (Input.GetKey(KeyCode.Q))
        {
            contActionsOut[2] -= 1f;
        }
        if (Input.GetKey(KeyCode.E))
        {
            contActionsOut[2] += 1f;
        }
        Debug.Log("Vertical: " + contActionsOut[0]);
        Debug.Log("Horizontal: " + contActionsOut[1]);
    }
    /// <summary>
    /// Used to provide a "kick" to the ball.
    /// </summary>
    void OnCollisionEnter(Collision c)
    {
        var force = k_Power * m_KickPower;
        if (position == Position.Goalie)
        {
            force = k_Power;
        }
        if (c.gameObject.CompareTag("ball"))
        {
            AddReward(.2f * m_BallTouch);
            var dir = c.contacts[0].point - transform.position;
            dir = dir.normalized;
            c.gameObject.GetComponent<Rigidbody>().AddForce(dir * force);
        }
    }

    public override void OnEpisodeBegin()
    {
        m_BallTouch = m_ResetParams.GetWithDefault("ball_touch", 0);
    }

}
