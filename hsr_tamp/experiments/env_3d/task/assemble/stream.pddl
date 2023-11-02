(define (stream hsr-assemble-tamp)
  (:stream sample-grasp
    ; sample grasp pose
    ; object ?o must be graspable and output ?g mush satisfy grasp condition
    :inputs (?o)
    :domain (Graspable ?o)
    :outputs (?g)
    :certified (Grasp ?o ?g)
  )

  (:stream sample-place
    ; sample place pose
    ; object ?o mush be placeable on the surface ?r and output mush be stable on the surface ?r at pose ?p
    :inputs (?b1 ?b2)
    :domain (Placeable ?b1 ?b2)
    :outputs (?p)
    :certified (and (Pose ?b1 ?p) (RegionPose ?b2 ?p))
  )

  (:stream sample-insert
    ; sample insert pose
    ; object ?o mush be placeable on the surface ?r and output mush be stable on the surface ?r at pose ?p
    :inputs (?b1 ?b2)
    :domain (Insertable ?b1 ?b2)
    :outputs (?p)
    :certified (and (Pose ?b1 ?p) (HolePose ?b2 ?p))
  )

  (:stream inverse-kinematics
    ; sample IK pose
    ; arm ?a mush be controllable and object ?o must be in hand ?g at pose ?p
    :inputs (?a ?b ?p ?g)
    :domain (and (Controllable ?a) (Pose ?b ?p) (Grasp ?b ?g))
    :outputs (?q ?t)
    :certified (and (BConf ?q) (ATraj ?t) (Kin ?a ?b ?p ?g ?q ?t))
  )

  (:stream plan-base-motion
    :inputs (?q1 ?q2)
    :domain (and (BConf ?q1) (BConf ?q2))
    :outputs (?t)
    :certified (and (BTraj ?t) (BaseMotion ?q1 ?t ?q2))
  )

  (:stream test-cfree-pose-pose
    :inputs (?b1 ?p1 ?b2 ?p2)
    :domain (and (Pose ?b1 ?p1) (Pose ?b2 ?p2))
    :certified (CFreePosePose ?b1 ?p1 ?b2 ?p2)
  )

  (:stream test-cfree-approach-pose
    :inputs (?b1 ?p1 ?g1 ?b2 ?p2)
    :domain (and (Pose ?b1 ?p1) (Grasp ?b1 ?g1) (Pose ?b2 ?p2))
    :certified (CFreeApproachPose ?b1 ?p1 ?g1 ?b2 ?p2)
  )

  (:stream test-cfree-traj-pose
    :inputs (?t ?b2 ?p2)
    :domain (and (ATraj ?t) (Pose ?b2 ?p2))
    :certified (CFreeTrajPose ?t ?b2 ?p2)
  )

  (:stream test-supported
    :inputs (?b1 ?p1 ?b2 ?p2)
    :domain (and (Pose ?b1 ?p1) (Pose ?b2 ?p2))
    :certified (Supported ?b1 ?p1)
  )

  (:stream test-inserted
    :inputs (?b1 ?p1 ?b2 ?p2)
    :domain (and (Pose ?b1 ?p1) (Pose ?b2 ?p2))
    :certified (Assembled ?b1 ?p1)
  )

  (:function (MoveCost ?t)
    (and (BTraj ?t))
  )

  ;(:predicate (TrajPoseCollision ?t ?o2 ?p2)
  ;  (and (BTraj ?t) (Pose ?o2 ?p2))
  ;)
  ;(:predicate (TrajArmCollision ?t ?a ?q)
  ;  (and (BTraj ?t) (AConf ?a ?q))
  ;)
  ;(:predicate (TrajGraspCollision ?t ?a ?o ?g)
  ;  (and (BTraj ?t) (Arm ?a) (Grasp ?o ?g))
  ;)
)