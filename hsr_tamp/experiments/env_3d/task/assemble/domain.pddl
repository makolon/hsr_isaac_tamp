(define (domain hsr-assemble-tamp)
  (:requirements :strips :equality)
  (:predicates
    ; Static predicates
    (Arm ?a)
    (Block ?b)
    (Region ?r)
    (Pose ?b ?p)
    (Grasp ?b ?g)
    (BTraj ?t)
    (ATraj ?t)
    (AConf ?q)
    (BConf ?q)
    (Kin ?a ?b ?p ?g ?q ?t)
    (BaseMotion ?q1 ?t ?q2)
    (ArmMotion ?a ?q1 ?t ?q2)
    (Controllable ?a)
    (Graspable ?b)
    (Placeable ?b ?r)
    (Insertable ?b ?r)
    (Stable ?b ?p ?r)
    (RegionPose ?r ?p)
    (HolePose ?r ?p)
    (TrajPoseCollision ?t ?b ?p)
    (TrajArmCollision ?t ?a ?q)
    (TrajGraspCollision ?t ?a ?b ?g)
    (CFreePosePose ?b ?p ?b2 ?p2)
    (CFreeApproachPose ?b ?p ?g ?b2 ?p2)
    (CFreeTrajPose ?t ?b2 ?p2)

    ; Fluent predicates
    (AtPose ?b ?p)
    (AtGrasp ?a ?b ?g)
    (AtAConf ?a ?q)
    (AtBConf ?q)
    (InitPose ?b ?p)    ; retired
    (PickPose ?b ?p)    ; retired
    (PlacePose ?b ?p)   ; retired
    (InsertPose ?b ?p)  ; retired
    (CanMove)
    (HandEmpty ?a)
    (Holding ?a ?b)
    (Placed ?b)
    (Inserted ?b)
    (Supported ?b ?p)   ; retired
    (Assembled ?b ?p)   ; retired

    ; Derived predicates
    (On ?o ?r)
    (InHole ?o ?r)
    (Holding ?a ?o)
    (UnsafePose ?o ?p)
    (UnsafeApproach ?o ?p ?g)
    (UnsafeATraj ?t)
    (UnsafeBTraj ?t)
  )

  (:functions
    (MoveCost ?t)
    (PickCost)
    (PlaceCost)
    (InsertCost)
  )

  (:action move_base
    :parameters (?q1 ?q2 ?t)
    :precondition (and (BaseMotion ?q1 ?t ?q2)
                       (AtBConf ?q1)
                       (CanMove)
                       (not (UnsafeBTraj ?t)))
    :effect (and (AtBConf ?q2)
                 (not (AtBConf ?q1))
                 (not (CanMove))
                 (not (= ?q1 ?q2))
                 (increase (total-cost) (MoveCost ?t)))
  )

  (:action move_arm
    :parameters (?q1 ?q2 ?t)
    :precondition (and (ArmMotion ?a ?q1 ?t ?q2)
                       (AtAConf ?a ?q1)
                       (CanMove)
                       (not (UnSafeATraj ?t)))
    :effect (and (AtAConf ?a ?q2)
                 (not (AtAConf ?a ?q1))
                 (not (CanMove))
                 (not (= ?q1 ?q2))
                 (increase (total-cost) (MoveCost ?t)))
  )

  (:action pick
    :parameters (?a ?b ?p ?g ?q ?t)
    :precondition (and (Kin ?a ?b ?p ?g ?q ?t)         ; kinematic constraints for pick pose ?p
                       (HandEmpty ?a)                  ; hand-empty constraints
                       (Graspable ?b)                  ; block ?b is graspable
                       (AtBConf ?q)                    ; base position is at conf ?q
                       (AtPose ?b ?p)                  ; block ?b pose is at pose ?p
                       (not (CanMove))                 ; cannot move
                       (not (UnsafeApproach ?b ?p ?g)) ; pose ?p with grasp ?g is not unsafe
                       (not (UnsafeATraj ?t)))         ; trajectory ?t is not unsafe
    :effect (and (CanMove)                             ; can move
                 (AtGrasp ?a ?b ?g)                    ; block ?b is at grasp ?g
                 (not (HandEmpty ?a))                  ; hand is not empty
                 (increase (total-cost) (PickCost)))   ; increase total-cost
  )

  (:action place
    :parameters (?a ?b1 ?b2 ?p ?g ?q ?t)
    :precondition (and (Kin ?a ?b1 ?p ?g ?q ?t)         ; kinematic constraints for place pose
                       (AtBConf ?q)                     ; base position is at conf ?q
                       (AtGrasp ?a ?b1 ?g)              ; at-grasp constraints
                       (Placeable ?b1 ?b2)              ; block b1 can place on block b2
                       (RegionPose ?b2 ?p)              ; region ?b2 is region-pose ?p
                       (not (CanMove))                  ; cannot move
                       (not (Placed ?b1))               ; block b1 is not placed yet
                       (not (AtPose ?b1 ?p))            ; block ?b1 is not at place pose ?p
                       (not (UnsafePose ?b1 ?p))        ; pose ?p is not unsafe
                       (not (UnsafeApproach ?b1 ?p ?g)) ; pose ?p with grasp ?g is not unsafe
                       (not (UnsafeATraj ?t)))          ; trajectory ?t is not unsafe
    :effect (and (AtPose ?b1 ?p)                        ; lock ?b1 is at place-pose ?p
                 (AtBConf ?q)                           ; base position is at conf ?q
                 (Placed ?b1)                           ; block ?b1 is placed
                 (increase (total-cost) (PlaceCost)))   ; increase total-cost
  )

  (:action insert
    :parameters (?a ?b1 ?b2 ?p1 ?p2 ?g ?q1 ?q2 ?t)
    :precondition (and (Kin ?a ?b1 ?p2 ?g ?q2 ?t)       ; kinematic constraints for insert pose
                       (AtBConf ?q1)                    ; base position is at conf ?q1
                       (AtGrasp ?a ?b1 ?g)              ; at-grasp constraints
                       (Placed ?b1)                     ; block ?b1 is placed already
                       (Inserted ?b2)                   ; hole ?b2 is inserted already
                       (Insertable ?b1 ?b2)             ; block ?b1 can insert into hole ?b2
                       (AtPose ?b1 ?p1)                 ; block ?b1 is at-pose place pose ?p1
                       (HolePose ?b2 ?p2)               ; hole ?b2 is hole-pose ?p2
                       (not (CanMove))                  ; cannot move
                       (not (AtPose ?b1 ?p2))           ; block ?b1 is not at insert-pose ?p2
                       (not (Inserted ?b1))             ; block ?b1 is not inserted yet
                       (not (UnsafePose ?b1 ?p2))       ; pose ?p2 is not unsafe
                       (not (UnsafeApproach ?b1 ?p2 ?g)); pose ?p2 with grasp ?g is not unsafe
                       (not (UnsafeATraj ?t)))          ; trajectory ?t is not unsafe
    :effect (and (CanMove)                              ; can move
                 (Inserted ?b1)                         ; block ?b1 is inserted
                 (AtBConf ?q1)                          ; base position is at conf ?q2
                 (AtPose ?b1 ?p2)                       ; block ?b1 is at insert-pose ?p2
                 (HandEmpty ?a)                         ; hand is empty
                 (increase (total-cost) (InsertCost)))  ; increase total-cost
  )

  (:derived (On ?b1 ?b2)
    (exists (?p) (and (Placed ?b1)
                      (AtPose ?b1 ?p)))
  )

  (:derived (InHole ?b1 ?b2)
    (exists (?p) (and (Inserted ?b1)
                      (AtPose ?b1 ?p)))
  )

  (:derived (Holding ?a ?b)
    (exists (?g) (and (Arm ?a) (Grasp ?b ?g)
                      (AtGrasp ?a ?b ?g)))
  )

  (:derived (UnsafePose ?b ?p)
    (exists (?b2 ?p2) (and (Pose ?b ?p) (Pose ?b2 ?p2)
                           (not (= ?b ?b2))
                           (not (CFreePosePose ?b ?p ?b2 ?p2))
                           (AtPose ?b2 ?p2)))
  )

  (:derived (UnsafeApproach ?b ?p ?g)
    (exists (?b2 ?p2) (and (Grasp ?b ?g)
                           (Pose ?b ?p)
                           (Pose ?b2 ?p2)
                           (not (= ?b ?b2))
                           (not (CFreeApproachPose ?b ?p ?g ?b2 ?p2))
                           (AtPose ?b2 ?p2)))
  )

  (:derived (UnsafeATraj ?t)
    (exists (?b2 ?p2) (and (ATraj ?t)
                           (Pose ?b2 ?p2)
                           (not (CFreeTrajPose ?t ?b2 ?p2))
                           (AtPose ?b2 ?p2)))
  )
  (:derived (UnsafeBTraj ?t)
    (or (exists (?o2 ?p2) (and (TrajPoseCollision ?t ?o2 ?p2)
                               (AtPose ?o2 ?p2)))
        (exists (?a ?q) (and (TrajArmCollision ?t ?a ?q)
                             (AtAConf ?a ?q)))
        (exists (?a ?o ?g) (and (TrajGraspCollision ?t ?a ?o ?g)
                                (AtGrasp ?a ?o ?g)))
    )
  )
)