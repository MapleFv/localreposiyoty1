package PracticePolymorphism;

public class LittleJoe extends Hero {

    public LittleJoe(int magicPoint, int hurt, String name) {
        super(magicPoint, hurt, name);
    }

    //攻击的方法
    public void attack() {
        System.out.println(this.getName() + "发动攻击，伤害为：" + this.getHurt() + ",消耗魔法值20！！！");
        this.setMagicPoint(getMagicPoint() - 20);

    }
    public void dazzlingStar(){
        System.out.println("释放大招：星华缭乱");
    }
}
