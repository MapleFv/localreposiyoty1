package PracticePolymorphism;

public class Daji extends Hero {

    public Daji(int magicPoint, int hurt, String name) {
        super(magicPoint, hurt, name);

    }

    public void attack() {
        System.out.println(this.getName() + "发动攻击，伤害为：" + this.getHurt() + ",消耗30的魔法值！！！");
        this.setMagicPoint(getMagicPoint() - 30);
    }
    public void queenWorship(){
        System.out.println("释放大招，女神崇拜！！！");
    }
}
