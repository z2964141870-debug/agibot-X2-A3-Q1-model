#include "writer_gate.hpp"
#include <cassert>
#include <iostream>
#include <limits>
using namespace x2_morph_backend;

WriterConfig Config() {
  WriterConfig c; c.kp.fill(50.); c.kd.fill(2.); c.lower.fill(-3.); c.upper.fill(3.);
  c.fault_kd.fill(2.); c.engage_s = .1; c.return_s = .1; return c;
}
WriterFeedback Feedback(double now) {
  WriterFeedback f; f.complete = true; f.gravity_z = -1.; f.oldest_receive_s = now-.001;
  f.q.fill(.2); return f;
}
MotorCommand Mc() { MotorCommand c; c.q.fill(.25); c.kp.fill(40.); c.kd.fill(3.); return c; }
WriterGate Armed(double now) {
  WriterGate g(Config()); Joints target; target.fill(.3);
  assert(g.Offer(target, 0, now, now)); assert(g.Arm(Feedback(now), now, Mc(), true));
  return g;
}
int main() {
  int cases = 0;
  { WriterGate g(Config()); assert(!g.Tick(Feedback(1),1)); assert(!g.Arm(Feedback(1),1,Mc(),true)); ++cases; }
  { WriterGate g(Config()); Joints q{}; assert(g.Offer(q,0,1,1)); assert(!g.Arm(Feedback(1),1,Mc(),false)); ++cases; }
  { auto g=Armed(1); auto c=g.Tick(Feedback(1),1); assert(c && c->q[0]==.2); ++cases; }
  { auto g=Armed(1); Joints q; q.fill(.3); assert(g.Offer(q,1,1.1,1.1));
    auto c=g.Tick(Feedback(1.1),1.1); assert(c && c->q==q); ++cases; }
  { auto g=Armed(1); auto c=g.Tick(Feedback(1.081),1.081); assert(c && c->kp[0]==0 && c->kd[0]==2);
    Joints q{}; assert(!g.Offer(q,1,1.082,1.082)); assert(!g.Arm(Feedback(1.082),1.082,Mc(),true)); ++cases; }
  { auto g=Armed(1); auto f=Feedback(1.01); f.oldest_receive_s=.5;
    assert(g.Tick(f,1.01)->kp[0]==0); assert(g.stage()==Stage::Fault); ++cases; }
  { auto g=Armed(1); auto f=Feedback(1.01); f.dq[8]=36.; assert(g.Tick(f,1.01)->kp[0]==0); ++cases; }
  { auto g=Armed(1); auto f=Feedback(1.01); f.gravity_z=0.; assert(g.Tick(f,1.01)->kp[0]==0); ++cases; }
  { auto g=Armed(1); Joints q{}; q[3]=std::numeric_limits<double>::quiet_NaN();
    assert(!g.Offer(q,1,1.01,1.01)); assert(g.Tick(Feedback(1.01),1.01)->kp[0]==0); ++cases; }
  { auto g=Armed(1); Joints q{}; q[3]=4.; assert(g.Offer(q,1,1.01,1.01)); ++cases; }
  { auto g=Armed(1); Joints q{}; assert(!g.Offer(q,0,1.01,1.01)); ++cases; }
  { auto g=Armed(1); Joints q{}; assert(!g.Offer(q,1,2.,1.01)); ++cases; }
  { auto g=Armed(1); g.Fault("transport_lost"); assert(g.stage()==Stage::Fault);
    assert(g.Tick(Feedback(1.01),1.01)->kp[0]==0); assert(!g.RequestReturn(1.01)); ++cases; }
  { auto g=Armed(1); g.Tick(Feedback(1),1); assert(g.RequestReturn(1.01));
    assert(g.Tick(Feedback(1.2),1.2)->q==Mc().q); assert(g.stage()==Stage::HoldForMc);
    assert(g.Tick(Feedback(1.21),1.21)->kp==Mc().kp);
    g.ExternalPublisher(); assert(!g.Tick(Feedback(1.22),1.22)); assert(g.stage()==Stage::Released); ++cases; }
  { auto g=Armed(1); g.ExternalPublisher(); assert(!g.Tick(Feedback(1.01),1.01));
    assert(g.reason()=="unexpected_external_writer"); ++cases; }
  { auto c=Config(); c.kd[0]=-1.; bool threw=false; try { WriterGate g(c); } catch(...) { threw=true; }
    assert(threw); ++cases; }
  { auto g=Armed(1); assert(g.RequestReturn(1.01)); g.EStop("operator_estop");
    assert(g.stage()==Stage::Fault); assert(g.Tick(Feedback(1.02),1.02)->kp[0]==0.); ++cases; }
  { auto g=Armed(1); assert(g.RequestReturn(1.01)); g.Tick(Feedback(1.2),1.2);
    g.EStop("control_transport_closed"); assert(g.stage()==Stage::Fault); ++cases; }
  { auto g=Armed(1); assert(g.RequestReturn(1.01));
    assert(g.Tick(Feedback(1.01),1.01)->q==Feedback(1).q); ++cases; }
  std::cout << "{\"status\":\"WRITER_GATE_PASSED\",\"cases\":" << cases << ",\"hardware_commands\":0}\n";
}
