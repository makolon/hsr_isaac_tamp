(define (stream gearbox2d-tamp)
  ; Sampling Stream
  (:stream s-region
    :inputs (?b ?r)
    :domain (and (Placeable ?b ?r) (Block ?b) (Region ?r))
    :outputs (?p)
    :certified (and (Pose ?b ?p) (Contain ?b ?p ?r)))

  (:stream s-grasp
    :inputs (?b)
    :domain (Block ?b)
    :outputs (?g)
    :certified (Grasp ?b ?g))

  (:stream s-ik
    :inputs (?b ?p ?g)
    :domain (and (Pose ?b ?p) (Grasp ?b ?g))
    :outputs (?q)
    :certified (and (Conf ?q) (Kin ?b ?q ?p ?g)))

  (:stream s-motion
    :inputs (?q1 ?q2)
    :domain (and (Conf ?q1) (Conf ?q2))
    :outputs (?t)
    :certified (and (Traj ?t) (Motion ?q1 ?t ?q2)))

  (:stream s-blockregion
    :inputs (?b1 ?b2 ?p2)
    :domain (and (Pose ?b2 ?p2) (Placeable ?b1 ?b2))
    :outputs (?p1)
    :certified (and (Pose ?b1 ?p1) (BlockContain ?b1 ?p1 ?b2 ?p2)))

  (:stream s-stackik
    :inputs (?b ?p ?g ?b2 ?p2)
    :domain (and (Pose ?b ?p) (Grasp ?b ?g) (Pose ?b2 ?p2))
    :outputs (?q)
    :certified (and (Conf ?q) (StackKin ?b ?q ?p ?g ?b2 ?p2)))

  ; Test Stream
  (:stream t-cfree
    :inputs (?b1 ?p1 ?b2 ?p2)
    :domain (and (Pose ?b1 ?p1) (Pose ?b2 ?p2))
    :certified (CFree ?b1 ?p1 ?b2 ?p2))

  (:stream t-region
    :inputs (?b ?p ?r)
    :domain (and (Pose ?b ?p) (Placeable ?b ?r) (Region ?r))
    :certified (Contain ?b ?p ?r))
  
  ;(:stream t-sfree
  ;  :inputs (?b1 ?p1 ?b2 ?p2)
  ;  :domain (and (Pose ?b1 ?p1) (Pose ?b2 ?p2))
  ;  :certified (SFree ?b1 ?p1 ?b2 ?p2))

  (:stream t-blockregion
    :inputs (?b1 ?p1 ?b2 ?p2)
    :domain (and (Pose ?b1 ?p1) (Placeable ?b1 ?b2) (Pose ?b2 ?p2) (Block ?b1) (Block ?b2))
    :certified (BlockContain ?b1 ?p1 ?b2 ?p2))

  (:stream t-cstack
    :inputs (?b1 ?p1 ?b2 ?p2)
    :domain (and (Pose ?b1 ?p1) (Pose ?b2 ?p2))
    :certified (CStack ?b1 ?p1 ?b2 ?p2))

  ; Function
  (:function (Dist ?q1 ?q2)
    (and (Conf ?q1) (Conf ?q2)))

  (:function (Duration ?t)
             (Traj ?t))
)