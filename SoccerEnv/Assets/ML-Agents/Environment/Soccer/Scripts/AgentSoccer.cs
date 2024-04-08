using UnityEngine;
using System.Collections.Generic;
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

    public Vector3 ownGoalPos;
    public Vector3 opposingGoalPos;

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
        SoccerEnvController envController = GetComponentInParent<SoccerEnvController>();

        // Get Positions in World Space
        var self_velocity = Vector3.Normalize(agentRb.velocity);
        var mate_pos = envController.getMate(this).transform.localPosition;
        var opponents = envController.getOpponents(this.team);
        var opponent_1_pos = opponents[0].transform.localPosition;
        var opponent_2_pos = opponents[1].transform.localPosition;
        var ball_pos = envController.ball.transform.localPosition;
        var ball_velocity = Vector3.Normalize(envController.ballRb.velocity);
        var ownGoalLocal = ownGoalPos;
        var opposingGoalLocal = opposingGoalPos;

        // Transform to Local Space
        Vector3[] transformedPos = new Vector3[] { mate_pos, opponent_1_pos, opponent_2_pos, ball_pos, ownGoalLocal, opposingGoalLocal };
        Vector3[] transformedVel = new Vector3[] { self_velocity, ball_velocity };
        transform.InverseTransformDirections(transformedVel);
        transform.InverseTransformPoints(transformedPos);

        // Register Observations
        sensor.AddObservation(transformedVel[0].x);
        sensor.AddObservation(transformedVel[0].z);
        sensor.AddObservation(transformedPos[0].x);
        sensor.AddObservation(transformedPos[0].z);
        sensor.AddObservation(transformedPos[1].x);
        sensor.AddObservation(transformedPos[1].z);
        sensor.AddObservation(transformedPos[2].x);
        sensor.AddObservation(transformedPos[2].z);
        sensor.AddObservation(transformedPos[3].x);
        sensor.AddObservation(transformedPos[3].z);
        sensor.AddObservation(transformedVel[1].x);
        sensor.AddObservation(transformedVel[1].z);
        sensor.AddObservation(transformedPos[4].x);
        sensor.AddObservation(transformedPos[4].z);
        sensor.AddObservation(transformedPos[5].x);
        sensor.AddObservation(transformedPos[5].z);
    }

    public void MoveAgent(ActionSegment<int> act)
    {
        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;

        m_KickPower = 0f;

        var forwardAxis = act[0];
        var rightAxis = act[1];
        var rotateAxis = act[2];

        switch (forwardAxis)
        {
            case 1:
                dirToGo = transform.forward * m_ForwardSpeed;
                m_KickPower = 1f;
                break;
            case 2:
                dirToGo = transform.forward * -m_ForwardSpeed;
                break;
        }

        switch (rightAxis)
        {
            case 1:
                dirToGo = transform.right * m_LateralSpeed;
                break;
            case 2:
                dirToGo = transform.right * -m_LateralSpeed;
                break;
        }

        switch (rotateAxis)
        {
            case 1:
                rotateDir = transform.up * -1f;
                break;
            case 2:
                rotateDir = transform.up * 1f;
                break;
        }

        transform.Rotate(rotateDir, Time.deltaTime * 100f);
        agentRb.AddForce(dirToGo * m_SoccerSettings.agentRunSpeed,
            ForceMode.VelocityChange);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        float actionX = Mathf.Clamp(actionBuffers.ContinuousActions[0], -1f, 1f);
        float actionZ = Mathf.Clamp(actionBuffers.ContinuousActions[1], -1f, 1f);
        float actionRotate = Mathf.Clamp(actionBuffers.ContinuousActions[2], -1f, 1f);
        m_KickPower = 0f;

        // Debug.Log("Action Received : [" + actionZ + ", " + actionX + ", " + actionRotate + "]");

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

        if (actionZ > 0)
        {
            m_KickPower = 1f;
        }

        transform.Rotate(transform.up * actionRotate, Time.deltaTime * 100f);
        agentRb.AddForce(transform.forward * m_ForwardSpeed * actionZ, ForceMode.VelocityChange);
        agentRb.AddForce(transform.right * m_LateralSpeed * actionX, ForceMode.VelocityChange);

        //MoveAgent(actionBuffers.DiscreteActions);
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
            var agent_ball = c.contacts[0].point - transform.position;
            agent_ball = agent_ball.normalized;
            c.gameObject.GetComponent<Rigidbody>().AddForce(agent_ball * force);
            // Reward if ball go to the goal
            Vector2 agent_goal = new Vector2(opposingGoalPos.x - transform.position.x, opposingGoalPos.z - transform.position.z);
            if (Vector2.Dot(agent_ball, agent_goal) >= 0)
            {
                //Debug.Log("Ball Touch Reward : " + .2f * m_BallTouch);
                AddReward(0.2f * m_BallTouch);
            }
            else
            {
                //Debug.Log("Ball Touch Reward : -" + 0.005f * m_BallTouch);
                AddReward(-0.01f * m_BallTouch);
            }
        }
    }

    public override void OnEpisodeBegin()
    {
        m_BallTouch = m_ResetParams.GetWithDefault("ball_touch", 0f);
    }

}
