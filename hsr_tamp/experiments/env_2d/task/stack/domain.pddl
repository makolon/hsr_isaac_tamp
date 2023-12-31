(define (domain gearbox2d-tamp)
  (:requirements :strips :equality)
  (:predicates
    ; Static predicates
    (Robot ?r)
    (Block ?b)
    (Region ?s)
    (Conf ?q)
    (Traj ?t)
    (Pose ?b ?p)
    (Grasp ?b ?g)
    (Motion ?q1 ?t ?q2)
    (Contain ?b ?p ?s)
    (BlockContain ?b1 ?p1 ?b2 ?p2)
    (Kin ?b ?q ?p ?g)
    (StackKin ?b ?q ?p ?g ?b2 ?p2)
    (Placeable ?b ?s)
    (Stackable ?b)
    (CStack ?b1 ?p1 ?b2 ?p2)
    (CFree ?b1 ?p1 ?b2 ?p2)
    (SFree ?b1 ?p1 ?b2 ?p2)
    (PoseCollision ?b1 ?p1 ?b2 ?p2)
    (TrajCollision ?t ?b2 ?p2)

    ; Fluent predicates
    (AtPose ?b ?p)
    (AtGrasp ?r ?b ?g)
    (AtConf ?r ?q)
    (Stacked ?b1 ?b2)
    (Holding ?r ?b)
    (HandEmpty ?r)
    (CanMove ?r)
    (CanManipulate ?r)

    ; Derived predicates
    (In ?b ?s)
    (On ?b1 ?b2)
    (UnsafePose ?b ?p)
    (UnsafeTraj ?t)
    (UnderBlock ?bl)
    (CollisionFree ?b ?p)
    (StackFree ?b ?p)
  )
  (:functions
    (Dist ?q1 ?q2)
  )

  (:action move
    :parameters (?r ?q1 ?t ?q2)
    :precondition (and (Robot ?r) (Motion ?q1 ?t ?q2)
                       (AtConf ?r ?q1) (CanMove ?r))
    :effect (and (AtConf ?r ?q2)
                 (not (AtConf ?r ?q1)) (not (CanMove ?r))
                 (increase (total-cost) (Dist ?q1 ?q2)))
  )

  (:action pick
    :parameters (?r ?b ?p ?g ?q)
    :precondition (and (Robot ?r) (Kin ?b ?q ?p ?g)
                       (AtConf ?r ?q) (AtPose ?b ?p) (HandEmpty ?r)
                       (not (UnderBlock ?b)))
    :effect (and (AtGrasp ?r ?b ?g) (CanMove ?r)
                 (not (AtPose ?b ?p)) (not (HandEmpty ?r))
                 (increase (total-cost) 10))
  )

  (:action place
   :parameters (?r ?b ?p ?g ?q)
   :precondition (and (Robot ?r) (Kin ?b ?q ?p ?g)
                      (AtConf ?r ?q) (AtGrasp ?r ?b ?g) ;(not (UnsafePose ?b ?p))
                      (forall (?b2 ?p2)
                        (imply (and (Pose ?b2 ?p2) (AtPose ?b2 ?p2))
                               (CFree ?b ?p ?b2 ?p2))))
   :effect (and (AtPose ?b ?p) (HandEmpty ?r) (CanMove ?r) (Stackable ?b)
                (not (AtGrasp ?r ?b ?g))
                (increase (total-cost) 10))
  )

  (:action stack
    :parameters (?r ?b1 ?p1 ?g ?q ?b2 ?p2)
    :precondition (and (Robot ?r) (Kin ?b1 ?q ?p1 ?g)
                       (AtConf ?r ?q) (AtGrasp ?r ?b1 ?g) (Stackable ?b2)
                       (Block ?b2) (AtPose ?b2 ?p2) (not (= ?b1 ?b2)))
    :effect (and (AtPose ?b1 ?p1) (HandEmpty ?r) (CanMove ?r) (Stacked ?b1 ?b2)
                 (not (AtGrasp ?r ?b1 ?g))
                 (increase (total-cost) 10))
  )

  (:derived (In ?b ?s)
    (exists (?p) (and (Contain ?b ?p ?s)
                      (AtPose ?b ?p))))

  (:derived (On ?b1 ?b2)
    (exists (?p1 ?p2) (and (BlockContain ?b1 ?p1 ?b2 ?p2)
                      (Block ?b1) (Block ?b2) 
                      (AtPose ?b1 ?p1) (AtPose ?b2 ?p2) (Stacked ?b1 ?b2))))

  (:derived (Holding ?r ?b)
    (exists (?g) (and (Robot ?r) (Grasp ?b ?g)
                      (AtGrasp ?r ?b ?g))))

  ;(:derived (UnderBlock ?b2)
  ;  (exists (?b1) (and (Block ?b1) (Block ?b2) (not (= ?b1 ?b2))
  ;                              (Stacked ?b1 ?b2))))

  ;(:derived (StackFree ?b ?p)
  ;  (forall (?b2 ?p2) (imply (and (Pose ?b2 ?p2) (AtPose ?b2 ?p2) (not (= ?b2 ?b)))
  ;                              (SFree ?b2 ?p2 ?b ?p) )))
)